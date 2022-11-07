class TemplateHandler():
  def __init__(self):
    self.labelmapper = {"OFF": "as the occurence of certain words has allowed us to detect whether a sentence is offensive or not, the given sentence is classified to be offensive", "NOT": "as the occurence of certain words has allowed us to detect whether a sentence is offensive or not, the given sentence is deemed not offensive"}
    self.explanation_filler = "because of word instances like"

  def explainer(self, lb, offensive_words):
    return f"{self.labelmapper[lb]} {self.explanation_filler} {', '.join(offensive_words)} ." if lb=="OFF" else f"{self.labelmapper[lb]}"

  def decode_onlylabel(self, sentence):
    sentence = sentence.lower() #.replace("the comment is ", "")
    label = sentence.split(self.explanation_filler)[0].lower().strip()
    if not label in self.labelmapper.values():
      label = "wrong-pred"

    return label
