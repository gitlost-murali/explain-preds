class TemplateHandler():
  def __init__(self):
    self.labelmapper = {"OFF": "the model observed classified offensive", "NOT": "the model observed classified not offensive"}
    self.explanation_filler = "since the following words showed up"

  def explainer(self, lb, offensive_words):
    return f"{self.labelmapper[lb]} {self.explanation_filler} {', '.join(offensive_words)} ." if lb=="OFF" else f"{self.labelmapper[lb]}"

  def decode_onlylabel(self, sentence):
    sentence = sentence.lower() #.replace("the comment is ", "")
    label = sentence.split(self.explanation_filler)[0].lower().strip()
    if not label in self.labelmapper.values():
      label = "wrong-pred"

    return label
