
class CtrClass:

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def debug(self, msg):
        self.constructLog(msg, "DEBUG")

    def info(self, msg):
        self.constructLog(msg, "INFO")

    def error(self, msg):
        self.constructLog(msg, "ERROR")

    def constructLog(self, msg, level):
        c_name = self.__class__.__name__
        s = f"[{c_name}] {level}:{msg}"
        self._log(s)

    def _log(self, msg):
        print(msg)