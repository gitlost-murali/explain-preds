class TemplateHandler():
  def __init__(self):
    self.labelmapper = {"OFF": "offensive", "NOT": "not offensive"}

  def decode_preds(self, sentence):
    sentence = sentence.lower()
    if not sentence in self.labelmapper.values():
      sentence = "wrong-pred"
    return sentence
