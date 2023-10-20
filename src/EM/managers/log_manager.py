import os
import time

from src.EM.utils import MessageAttribute, mkdir


class LogManager(object):
    def __init__(self, root_path: str, log_to_disk: bool = False):
        self.root_path = root_path
        self.log_path = None
        self.log_to_disk = log_to_disk
        self.writer = None

        self._file_ptr = None

        self.initialized = False

    def Initialization(self):
        if self.NeedLogToDisk():
            root_path = self.GetRootPath()
            mkdir(os.path.join(root_path, "log"))

            str_curr_time = time.strftime(
                "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
            )

            self._file_ptr = open(
                os.path.join(root_path, "log", str("Log-") + str_curr_time + ".txt"),
                "w",
            )
            self._file_ptr.close()

            self.log_path = os.path.join(
                root_path, "log", str("Log-") + str_curr_time + ".txt"
            )

            self._file_ptr = open(self.log_path, "a+")

            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(os.path.join(root_path, "log"))

        self.initialized = True

    def GetRootPath(self):
        return self.root_path

    def NeedLogToDisk(self):
        return self.log_to_disk

    def WriterAddScalar(self, *args, **kwargs):
        if self.NeedLogToDisk():
            self.writer.add_scalar(*args, **kwargs)

    def WarnLog(self, sentences=""):
        self._Slog(message_attribute=MessageAttribute.EWarn, sentences=sentences)

    def InfoLog(self, sentences=""):
        self._Slog(message_attribute=MessageAttribute.EInfo, sentences=sentences)

    def ErrorLog(self, sentences=""):
        self._Slog(message_attribute=MessageAttribute.EError, sentences=sentences)
        raise RuntimeError()

    def _Slog(self, message_attribute=MessageAttribute(0), sentences=""):
        str_curr_time = time.strftime(
            "[%Y-%m-%d %H:%M:%S] (", time.localtime(time.time())
        )
        prefix_str = ""
        final_str = ""
        if message_attribute.value == MessageAttribute.EWarn.value:
            # Set Font as yellow
            prefix_str = "\033[1;33m"
            final_str = "\033[0m"
        elif message_attribute.value == MessageAttribute.EError.value:
            # Set Font as Red
            prefix_str = "\033[1;31m"
            final_str = "\033[0m"
        elif message_attribute.value == MessageAttribute.EInfo.value:
            # Set Font as Red
            prefix_str = "\033[1;32m"
            final_str = "\033[0m"
        print(
            prefix_str
            + str_curr_time
            + message_attribute.name
            + ") "
            + sentences
            + final_str
        )
        if self.NeedLogToDisk() and self.initialized:
            self._file_ptr.write(
                str_curr_time + message_attribute.name + ") " + sentences + "\n"
            )

    def kill(self):
        self._Slog(MessageAttribute.EError, sentences="Kill the Log Manager")
        if self.NeedLogToDisk() and self.initialized:
            if self._file_ptr is not None:
                self._file_ptr.close()
