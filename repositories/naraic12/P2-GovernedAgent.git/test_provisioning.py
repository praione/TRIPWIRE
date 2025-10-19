from auto_provisioning_system import AutoProvisioningSystem

provisioner = AutoProvisioningSystem()
status = provisioner.get_provisioning_status()
print('Status:', status)
print('Circuit breakers:', status['circuit_breakers'])
