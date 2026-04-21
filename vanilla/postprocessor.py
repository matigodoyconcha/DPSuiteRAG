from core.postprocessor import BasePostprocessor

class IdentityPostprocessor(BasePostprocessor):
    def process(self, response: str) -> str:
        return response
