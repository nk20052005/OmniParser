"""
Azure VM operation tools.

Uses Azure SDK for Python to manage virtual machines.
"""

import logging
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class AzureVMStartTool(BaseTool):
    name = "azure_vm_start"
    description = "Start (power on) an Azure virtual machine"
    required_params = ["vm_name"]
    optional_params = ["resource_group", "subscription"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        vm_name = params["vm_name"]
        rg = params.get("resource_group", self._default_rg)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient

            credential = DefaultAzureCredential()
            client = ComputeManagementClient(credential, self._subscription_id)
            poller = client.virtual_machines.begin_start(rg, vm_name)
            poller.result()
            return ToolResult(
                success=True,
                data={"vm_name": vm_name, "status": "starting", "resource_group": rg},
                message=f"VM {vm_name} is starting.",
            )
        except Exception as e:
            logger.exception("Failed to start VM %s", vm_name)
            return ToolResult(success=False, error=str(e))


class AzureVMStopTool(BaseTool):
    name = "azure_vm_stop"
    description = "Stop (power off) an Azure virtual machine"
    required_params = ["vm_name"]
    optional_params = ["resource_group", "force"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        vm_name = params["vm_name"]
        rg = params.get("resource_group", self._default_rg)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient

            credential = DefaultAzureCredential()
            client = ComputeManagementClient(credential, self._subscription_id)
            poller = client.virtual_machines.begin_power_off(rg, vm_name)
            poller.result()
            return ToolResult(
                success=True,
                data={"vm_name": vm_name, "status": "stopping", "resource_group": rg},
                message=f"VM {vm_name} is stopping.",
            )
        except Exception as e:
            logger.exception("Failed to stop VM %s", vm_name)
            return ToolResult(success=False, error=str(e))


class AzureVMRestartTool(BaseTool):
    name = "azure_vm_restart"
    description = "Restart an Azure virtual machine"
    required_params = ["vm_name"]
    optional_params = ["resource_group"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        vm_name = params["vm_name"]
        rg = params.get("resource_group", self._default_rg)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient

            credential = DefaultAzureCredential()
            client = ComputeManagementClient(credential, self._subscription_id)
            poller = client.virtual_machines.begin_restart(rg, vm_name)
            poller.result()
            return ToolResult(
                success=True,
                data={"vm_name": vm_name, "status": "restarting", "resource_group": rg},
                message=f"VM {vm_name} is restarting.",
            )
        except Exception as e:
            logger.exception("Failed to restart VM %s", vm_name)
            return ToolResult(success=False, error=str(e))


class AzureVMResizeTool(BaseTool):
    name = "azure_vm_resize"
    description = "Resize an Azure virtual machine to a new SKU"
    required_params = ["vm_name", "new_size"]
    optional_params = ["resource_group"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        vm_name = params["vm_name"]
        new_size = params["new_size"]
        rg = params.get("resource_group", self._default_rg)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient

            credential = DefaultAzureCredential()
            client = ComputeManagementClient(credential, self._subscription_id)
            vm = client.virtual_machines.get(rg, vm_name)
            vm.hardware_profile.vm_size = new_size
            poller = client.virtual_machines.begin_create_or_update(rg, vm_name, vm)
            poller.result()
            return ToolResult(
                success=True,
                data={"vm_name": vm_name, "new_size": new_size, "resource_group": rg},
                message=f"VM {vm_name} resized to {new_size}.",
            )
        except Exception as e:
            logger.exception("Failed to resize VM %s", vm_name)
            return ToolResult(success=False, error=str(e))


class AzureVMDeallocateTool(BaseTool):
    name = "azure_vm_deallocate"
    description = "Deallocate an Azure virtual machine (release compute resources)"
    required_params = ["vm_name"]
    optional_params = ["resource_group"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        vm_name = params["vm_name"]
        rg = params.get("resource_group", self._default_rg)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient

            credential = DefaultAzureCredential()
            client = ComputeManagementClient(credential, self._subscription_id)
            poller = client.virtual_machines.begin_deallocate(rg, vm_name)
            poller.result()
            return ToolResult(
                success=True,
                data={"vm_name": vm_name, "status": "deallocated", "resource_group": rg},
                message=f"VM {vm_name} has been deallocated.",
            )
        except Exception as e:
            logger.exception("Failed to deallocate VM %s", vm_name)
            return ToolResult(success=False, error=str(e))


class AzureVMStatusTool(BaseTool):
    name = "azure_vm_status"
    description = "Get the current status of an Azure virtual machine"
    required_params = ["vm_name"]
    optional_params = ["resource_group"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        vm_name = params["vm_name"]
        rg = params.get("resource_group", self._default_rg)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient

            credential = DefaultAzureCredential()
            client = ComputeManagementClient(credential, self._subscription_id)
            vm = client.virtual_machines.get(rg, vm_name, expand="instanceView")
            statuses = vm.instance_view.statuses if vm.instance_view else []
            power_state = "unknown"
            for s in statuses:
                if s.code and s.code.startswith("PowerState/"):
                    power_state = s.code.replace("PowerState/", "")
                    break
            return ToolResult(
                success=True,
                data={
                    "vm_name": vm_name,
                    "power_state": power_state,
                    "vm_size": vm.hardware_profile.vm_size if vm.hardware_profile else "unknown",
                    "location": vm.location,
                    "resource_group": rg,
                },
            )
        except Exception as e:
            logger.exception("Failed to get status for VM %s", vm_name)
            return ToolResult(success=False, error=str(e))


class AzureVMListTool(BaseTool):
    name = "azure_vm_list"
    description = "List Azure virtual machines"
    required_params: list[str] = []
    optional_params = ["resource_group", "status_filter"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        rg = params.get("resource_group", self._default_rg)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient

            credential = DefaultAzureCredential()
            client = ComputeManagementClient(credential, self._subscription_id)

            if rg:
                vms = client.virtual_machines.list(rg)
            else:
                vms = client.virtual_machines.list_all()

            vm_list = []
            for vm in vms:
                vm_list.append({
                    "name": vm.name,
                    "location": vm.location,
                    "vm_size": vm.hardware_profile.vm_size if vm.hardware_profile else "unknown",
                    "resource_group": vm.id.split("/")[4] if vm.id else "",
                })

            status_filter = params.get("status_filter")
            # Note: status filtering would require instance view calls
            return ToolResult(
                success=True,
                data={"vms": vm_list, "count": len(vm_list)},
            )
        except Exception as e:
            logger.exception("Failed to list VMs")
            return ToolResult(success=False, error=str(e))


class AzureVMMetricsTool(BaseTool):
    name = "azure_vm_metrics"
    description = "Get performance metrics for an Azure virtual machine"
    required_params = ["vm_name"]
    optional_params = ["resource_group", "metric_type", "time_range"]

    def __init__(self, subscription_id: str = "", resource_group: str = ""):
        self._subscription_id = subscription_id
        self._default_rg = resource_group

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        vm_name = params["vm_name"]
        rg = params.get("resource_group", self._default_rg)
        metric_type = params.get("metric_type", "cpu")
        try:
            from datetime import datetime, timedelta

            from azure.identity import DefaultAzureCredential
            from azure.mgmt.monitor import MonitorManagementClient

            credential = DefaultAzureCredential()
            monitor_client = MonitorManagementClient(credential, self._subscription_id)

            resource_id = (
                f"/subscriptions/{self._subscription_id}/resourceGroups/{rg}"
                f"/providers/Microsoft.Compute/virtualMachines/{vm_name}"
            )

            metric_map = {
                "cpu": "Percentage CPU",
                "memory": "Available Memory Bytes",
                "disk": "Disk Read Bytes,Disk Write Bytes",
                "network": "Network In Total,Network Out Total",
            }
            metric_name = metric_map.get(metric_type, "Percentage CPU")

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            metrics = monitor_client.metrics.list(
                resource_id,
                metricnames=metric_name,
                timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
                aggregation="Average",
            )

            result_data = []
            for metric in metrics.value:
                for ts in metric.timeseries:
                    for dp in ts.data:
                        if dp.average is not None:
                            result_data.append({
                                "timestamp": dp.time_stamp.isoformat(),
                                "average": dp.average,
                            })

            return ToolResult(
                success=True,
                data={
                    "vm_name": vm_name,
                    "metric_type": metric_type,
                    "metric_name": metric_name,
                    "data_points": result_data[-10:],  # Last 10 points
                },
            )
        except Exception as e:
            logger.exception("Failed to get metrics for VM %s", vm_name)
            return ToolResult(success=False, error=str(e))


def create_azure_vm_tools(subscription_id: str = "", resource_group: str = "") -> list[BaseTool]:
    """Factory function to create all Azure VM tools."""
    return [
        AzureVMStartTool(subscription_id, resource_group),
        AzureVMStopTool(subscription_id, resource_group),
        AzureVMRestartTool(subscription_id, resource_group),
        AzureVMResizeTool(subscription_id, resource_group),
        AzureVMDeallocateTool(subscription_id, resource_group),
        AzureVMStatusTool(subscription_id, resource_group),
        AzureVMListTool(subscription_id, resource_group),
        AzureVMMetricsTool(subscription_id, resource_group),
    ]
