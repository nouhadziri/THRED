from tensorflow.python.client import device_lib

from .misc import safe_mod


class DeviceManager(object):
    """
    Derived from https://gist.github.com/jovianlin/b5b2c4be45437992df58a7dd13dbafa7
    """

    def __init__(self):
        local_device_protos = device_lib.list_local_devices()
        self.cpus = []
        self.gpus = []
        for dev in local_device_protos:
            if dev.device_type == 'GPU':
                self.gpus.append(dev.name)
            elif dev.device_type == 'CPU':
                self.cpus.append(dev.name)

    def get_default_device(self):
        if self.gpus:
            return self.gpus[0]

        return self.cpus[0]

    def num_available_gpus(self):
        return len(self.gpus)

    def gpu(self, index):
        return self.gpus[index] if self.gpus else self.get_default_device()

    def tail_gpu(self):
        return self.gpus[-1] if self.gpus else self.get_default_device()


class RoundRobin(object):

    def __init__(self, device_manager):
        self.device_manager = device_manager

    def assign(self, n_devices, base=0):
        devices = []

        for i in range(n_devices):
            devices.append(self.device_manager.gpu(safe_mod((base + i), self.device_manager.num_available_gpus())))

        return devices


if __name__ == "__main__":
    manager = DeviceManager()
    print(manager.gpus)

    round_robin = RoundRobin(manager)
    print(round_robin.assign(4))
