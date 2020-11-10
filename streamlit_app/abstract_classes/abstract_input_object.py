from abc import ABC, abstractmethod


class AbstractInputObject(ABC):

    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @abstractmethod
    def action(self):
        raise NotImplementedError()
