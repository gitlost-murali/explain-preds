class TemplateHandler():
  def __init__(self):
    self.labelmapper = {"OFF": "0", "NOT": "1"}

  def decode_preds(self, sentence):
    sentence = sentence.lower()
    if not sentence in self.labelmapper.values():
      sentence = "wrong-pred"
    return sentence
