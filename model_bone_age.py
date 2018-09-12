from bone_age import inference_bone_age

import argparse


class Model(object):

    def __init__(self, args):
        pass

    def predict(self, img):
        pass


class BoneAgePredictor(Model):

    def __init__(self, args):
        self.args = None
        self.model= inference_bone_age.load_model(args.params)
    def predict(self, img, gender):
        return inference_bone_age.predict(img, gender, self.model, self.args)

