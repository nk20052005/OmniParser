"""
Main Conversation Orchestrator.

Implements the full conversation pipeline:
  User Message → Intent Detection → Entity Extraction → Parameter Validation
  → Missing Info Check → Follow-Up Questions → Tool Selection → Execution
  → Response Generation → Memory Update
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .gemma_client import GemmaClient
from .intent import IntentCategory, IntentDetector, IntentResult
from .entities import EntityExtractor, ExtractedEntities
from .slots import SlotFiller, SlotValidationResult
from .response import ResponseGenerator
from ..memory.store import MemoryStore
from ..tools.registry import ToolRegistry
from ..approval.manager import ApprovalManager, ApprovalStatus

logger = logging.getLogger(__name__)


class ConversationState(str, Enum):
    """State of the current conversation turn."""
    IDLE = "idle"
    AWAITING_SLOT = "awaiting_slot"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"


@dataclass
class ConversationContext:
    """Tracks the state of an ongoing conversation."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    channel_id: str = ""
    state: ConversationState = ConversationState.IDLE
    current_intent: Optional[IntentCategory] = None
    collected_entities: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    pending_tool_call: Optional[dict] = None
    turn_count: int = 0

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
        self.turn_count += 1

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})

    def reset_action(self):
        """Reset the current action state but keep conversation history."""
        self.state = ConversationState.IDLE
        self.current_intent = None
        self.collected_entities = {}
        self.pending_tool_call = None


@dataclass
class ConversationResponse:
    """Response from the conversation engine."""
    message: str
    intent: Optional[IntentCategory] = None
    tool_called: Optional[str] = None
    tool_result: Optional[dict] = None
    awaiting_input: bool = False
    awaiting_approval: bool = False
    conversation_id: str = ""


class ConversationEngine:
    """
    Main orchestrator implementing the full conversation pipeline.

    Every message flows through:
    1. Intent Detection
    2. Entity Extraction
    3. Parameter Validation / Slot Filling
    4. Tool Selection & Execution (with approval if needed)
    5. Response Generation
    6. Memory Update
    """

    def __init__(
        self,
        gemma: GemmaClient,
        tool_registry: ToolRegistry,
        memory: MemoryStore,
        approval_manager: ApprovalManager,
    ):
        self._gemma = gemma
        self._intent_detector = IntentDetector(gemma)
        self._entity_extractor = EntityExtractor(gemma)
        self._slot_filler = SlotFiller(gemma)
        self._response_generator = ResponseGenerator(gemma)
        self._tools = tool_registry
        self._memory = memory
        self._approval = approval_manager

        # Active conversations keyed by (user_id, channel_id)
        self._conversations: dict[str, ConversationContext] = {}

    def _get_context(self, user_id: str, channel_id: str) -> ConversationContext:
        """Get or create conversation context for a user/channel pair."""
        key = f"{user_id}:{channel_id}"
        if key not in self._conversations:
            ctx = ConversationContext(user_id=user_id, channel_id=channel_id)
            self._conversations[key] = ctx
        return self._conversations[key]

    async def process_message(
        self,
        user_message: str,
        user_id: str,
        channel_id: str = "default",
    ) -> ConversationResponse:
        """
        Process an incoming user message through the full pipeline.
        """
        ctx = self._get_context(user_id, channel_id)
        ctx.add_user_message(user_message)

        try:
            # Load user memory/preferences for context enrichment
            user_prefs = self._memory.get_user_preferences(user_id)

            # --- STEP 1: Handle in-progress states ---
            if ctx.state == ConversationState.AWAITING_APPROVAL:
                return await self._handle_approval_response(ctx, user_message)

            if ctx.state == ConversationState.AWAITING_SLOT:
                return await self._handle_slot_response(ctx, user_message)

            # --- STEP 2: Intent Detection ---
            intent_result = self._intent_detector.detect(
                user_message, ctx.history
            )
            logger.info(
                "Intent detected: %s (confidence=%.2f, reason=%s)",
                intent_result.intent.value,
                intent_result.confidence,
                intent_result.reasoning,
            )

            # Handle general chat directly
            if intent_result.intent == IntentCategory.GENERAL_CHAT:
                return self._handle_general_chat(ctx, user_message)

            # Handle confirmation/rejection for any pending action
            if intent_result.intent == IntentCategory.CONFIRMATION:
                if ctx.pending_tool_call:
                    return await self._execute_tool(ctx)
                return self._handle_general_chat(ctx, user_message)

            if intent_result.intent == IntentCategory.REJECTION:
                ctx.reset_action()
                msg = "Understood, I've cancelled that action. What else can I help with?"
                ctx.add_assistant_message(msg)
                return ConversationResponse(
                    message=msg,
                    intent=IntentCategory.REJECTION,
                    conversation_id=ctx.conversation_id,
                )

            # --- STEP 3: Entity Extraction ---
            ctx.current_intent = intent_result.intent
            extracted = self._entity_extractor.extract(
                user_message, intent_result.intent, ctx.history
            )
            logger.info("Entities extracted: %s", extracted.entities)

            # Merge with any previously collected entities
            ctx.collected_entities = self._slot_filler.merge_entities(
                ctx.collected_entities, extracted.entities
            )

            # Enrich from memory (e.g., frequently used VMs)
            ctx.collected_entities = self._enrich_from_memory(
                ctx.collected_entities, intent_result.intent, user_id, user_prefs
            )

            # --- STEP 4: Slot Validation ---
            validation = self._slot_filler.validate(
                intent_result.intent,
                ctx.collected_entities,
                ctx.history,
            )

            if not validation.is_complete:
                # Need more information
                ctx.state = ConversationState.AWAITING_SLOT
                msg = validation.follow_up_question or "I need more information to proceed."
                ctx.add_assistant_message(msg)
                return ConversationResponse(
                    message=msg,
                    intent=intent_result.intent,
                    awaiting_input=True,
                    conversation_id=ctx.conversation_id,
                )

            # All slots filled — prepare tool call
            ctx.collected_entities = validation.filled_slots
            tool_name = self._tools.get_tool_for_intent(intent_result.intent)
            ctx.pending_tool_call = {
                "tool": tool_name,
                "parameters": ctx.collected_entities,
                "intent": intent_result.intent.value,
            }

            # --- STEP 5: Check if approval is needed ---
            if self._approval.requires_approval(intent_result.intent, ctx.collected_entities):
                ctx.state = ConversationState.AWAITING_APPROVAL
                msg = self._response_generator.generate_approval_request(
                    intent_result.intent, ctx.collected_entities
                )
                ctx.add_assistant_message(msg)
                return ConversationResponse(
                    message=msg,
                    intent=intent_result.intent,
                    awaiting_approval=True,
                    conversation_id=ctx.conversation_id,
                )

            # --- STEP 6: Execute ---
            return await self._execute_tool(ctx)

        except Exception as e:
            logger.exception("Error processing message")
            msg = self._response_generator.generate_error_response(str(e))
            ctx.add_assistant_message(msg)
            ctx.reset_action()
            return ConversationResponse(
                message=msg,
                conversation_id=ctx.conversation_id,
            )

    async def _handle_slot_response(
        self, ctx: ConversationContext, user_message: str
    ) -> ConversationResponse:
        """Handle a response when we were waiting for slot information."""
        if ctx.current_intent is None:
            ctx.reset_action()
            return self._handle_general_chat(ctx, user_message)

        # Check if the user wants to cancel
        cancel_check = self._intent_detector.detect(user_message, ctx.history)
        if cancel_check.intent == IntentCategory.REJECTION:
            ctx.reset_action()
            msg = "No problem, I've cancelled that. What else can I help with?"
            ctx.add_assistant_message(msg)
            return ConversationResponse(
                message=msg, intent=IntentCategory.REJECTION,
                conversation_id=ctx.conversation_id,
            )

        # Extract entities from the follow-up response
        extracted = self._entity_extractor.extract(
            user_message, ctx.current_intent, ctx.history
        )
        ctx.collected_entities = self._slot_filler.merge_entities(
            ctx.collected_entities, extracted.entities
        )

        # Re-validate
        validation = self._slot_filler.validate(
            ctx.current_intent, ctx.collected_entities, ctx.history
        )

        if not validation.is_complete:
            msg = validation.follow_up_question or "I still need more information."
            ctx.add_assistant_message(msg)
            return ConversationResponse(
                message=msg,
                intent=ctx.current_intent,
                awaiting_input=True,
                conversation_id=ctx.conversation_id,
            )

        # Slots are complete now
        ctx.collected_entities = validation.filled_slots
        tool_name = self._tools.get_tool_for_intent(ctx.current_intent)
        ctx.pending_tool_call = {
            "tool": tool_name,
            "parameters": ctx.collected_entities,
            "intent": ctx.current_intent.value,
        }

        # Check approval
        if self._approval.requires_approval(ctx.current_intent, ctx.collected_entities):
            ctx.state = ConversationState.AWAITING_APPROVAL
            msg = self._response_generator.generate_approval_request(
                ctx.current_intent, ctx.collected_entities
            )
            ctx.add_assistant_message(msg)
            return ConversationResponse(
                message=msg,
                intent=ctx.current_intent,
                awaiting_approval=True,
                conversation_id=ctx.conversation_id,
            )

        return await self._execute_tool(ctx)

    async def _handle_approval_response(
        self, ctx: ConversationContext, user_message: str
    ) -> ConversationResponse:
        """Handle user response to an approval request."""
        approval_intent = self._intent_detector.detect(user_message, ctx.history)

        if approval_intent.intent == IntentCategory.CONFIRMATION:
            return await self._execute_tool(ctx)
        elif approval_intent.intent == IntentCategory.REJECTION:
            ctx.reset_action()
            msg = "Action cancelled. Let me know if you need anything else."
            ctx.add_assistant_message(msg)
            return ConversationResponse(
                message=msg,
                intent=IntentCategory.REJECTION,
                conversation_id=ctx.conversation_id,
            )
        else:
            msg = "Please confirm with 'yes' or cancel with 'no'."
            ctx.add_assistant_message(msg)
            return ConversationResponse(
                message=msg,
                awaiting_approval=True,
                conversation_id=ctx.conversation_id,
            )

    async def _execute_tool(self, ctx: ConversationContext) -> ConversationResponse:
        """Execute the pending tool call and generate a response."""
        if not ctx.pending_tool_call:
            ctx.reset_action()
            return ConversationResponse(
                message="No pending action to execute.",
                conversation_id=ctx.conversation_id,
            )

        tool_name = ctx.pending_tool_call["tool"]
        parameters = ctx.pending_tool_call["parameters"]
        intent = IntentCategory(ctx.pending_tool_call["intent"])

        ctx.state = ConversationState.EXECUTING
        logger.info("Executing tool: %s with params: %s", tool_name, parameters)

        # Execute the tool
        tool_result = await self._tools.execute(tool_name, parameters)

        # Generate response
        msg = self._response_generator.generate(
            intent, tool_name, tool_result, parameters, ctx.history
        )

        # Update memory
        self._memory.record_action(
            user_id=ctx.user_id,
            intent=intent.value,
            tool=tool_name,
            parameters=parameters,
            result=tool_result,
        )

        ctx.add_assistant_message(msg)
        ctx.reset_action()

        return ConversationResponse(
            message=msg,
            intent=intent,
            tool_called=tool_name,
            tool_result=tool_result,
            conversation_id=ctx.conversation_id,
        )

    def _handle_general_chat(
        self, ctx: ConversationContext, user_message: str
    ) -> ConversationResponse:
        """Handle general chat messages (no tool invocation)."""
        msg = self._response_generator.generate_general_response(
            user_message, ctx.history
        )
        ctx.add_assistant_message(msg)
        return ConversationResponse(
            message=msg,
            intent=IntentCategory.GENERAL_CHAT,
            conversation_id=ctx.conversation_id,
        )

    def _enrich_from_memory(
        self,
        entities: dict[str, Any],
        intent: IntentCategory,
        user_id: str,
        user_prefs: dict,
    ) -> dict[str, Any]:
        """Enrich entities using memory (e.g., frequently used VMs)."""
        enriched = dict(entities)

        # If no VM name specified but user has a frequently used VM
        vm_intents = {
            IntentCategory.VM_START, IntentCategory.VM_STOP, IntentCategory.VM_RESTART,
            IntentCategory.VM_STATUS, IntentCategory.VM_METRICS, IntentCategory.VM_RESIZE,
            IntentCategory.VM_DEALLOCATE,
        }
        if intent in vm_intents and "vm_name" not in enriched:
            freq_vm = user_prefs.get("frequent_vm")
            if freq_vm:
                # Don't auto-fill, but the slot filler can suggest it
                pass  # Memory suggestion handled in slot filler

        return enriched

    def reset_conversation(self, user_id: str, channel_id: str = "default"):
        """Reset a conversation context."""
        key = f"{user_id}:{channel_id}"
        if key in self._conversations:
            del self._conversations[key]
