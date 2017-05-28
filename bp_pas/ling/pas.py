class PAS(object):

    def __init__(self, pred, args):
        self.pred = pred
        self.args = args

    def prd_index(self):
        return self.pred.word_index

    def __str__(self):
        pstr = "{}({}) -> ".format(self.pred.word_form,
                                   self.pred.word_index)
        for arg in self.args:
            pstr += "{}({}, {})".format(arg.word_form,
                                        arg.word_index,
                                        arg.arg_type)
        return pstr


class Predicate(object):

    def __init__(self, word_index, word_form):
        self.word_index = word_index
        self.word_form = word_form


    def __eq__(self, other):
        return self.word_index == other.word_index and self.word_form == other.word_form


class Argument(object):

    def __init__(self, word_index, word_form, arg_type):
        self.word_index = word_index
        self.word_form = word_form
        self.arg_type = arg_type

    def __eq__(self, other):
        return self.word_index == other.word_index and self.arg_type == other.arg_type

    def __str__(self):
        return "-> {}({},{})".format(self.word_form, self.word_index, self.arg_type)
