import numpy as np

from abc import ABCMeta, abstractmethod
from bp_pas.ling.word import BST, DEP, INTRA_ZERO, ConllWord
from bp_pas.ling.pas import PAS, Argument
from bp_pas.ling.sent import Sentence
import sys
from collections import defaultdict
from collections import Counter

def evaluate(test_pases, gold_pases, verbose=False):
    case_counts = defaultdict(lambda: Counter())
    total_correct = 0
    total_test = 0
    total_gold = 0
    for tpas, gpas in zip(test_pases, gold_pases):
#        print(tpas.pred.word_form)
#        print(gpas.pred.word_form)
        assert tpas.pred == gpas.pred
        for arg in tpas.args:
            if arg in gpas.args:
                total_correct += 1
            elif verbose:

                print('Predicted {}({}) --{}--> {}({})'.format(tpas.pred.word_form,
                                                                   tpas.pred.word_index,
                                                                   arg.arg_type,
                                                                   arg.word_form,
                                                                   arg.word_index))
                correction_lst =  [gold_arg for gold_arg in gpas.args if arg.word_index == gold_arg.word_index]
                if correction_lst:
                    print('  should be: --> {}'.format(correction_lst[0].arg_type))
                else:
                    print('  It should have been NIL')
                desired_arg_lst = [gold_arg for gold_arg in gpas.args if arg.arg_type == gold_arg.arg_type]
                for gold_arg in desired_arg_lst:
                    print('  Gold {} relation: --> {}({})'.format(arg.arg_type, gold_arg.word_form, gold_arg.word_index))
                print()

        total_test += len(tpas.args)
        total_gold += len(gpas.args)
    print('Prec = ({}/{}) = {}'.format(total_correct, total_test, total_correct * 100.0 / total_test))
    print('Rec  = ({}/{}) = {}'.format(total_correct, total_gold, total_correct * 100.0 / total_gold))


def evaluate_instance(test_pas, gold_pas):
    pass




#
# GA_ID = 1
# O_ID = 2
# NI_ID = 3
#
# def say(s, stream=sys.stdout):
#     stream.write(s)
#     stream.flush()
#
# #def ntc2pas(ntc_sent):
# #    prd_idx = -1
# #    args = []
# #    for w in ntc_sent:
# #        if w.case_index:
# #            args.append(Argument(w.index, , w.case_index, w.case_type))
# #        if w.is_prd:
# #            prd_idx = w.index
# #    assert prd_idx > -1
# #    pas = PAS()
#
# def ntc_sent2conll(ntc_words, PRD='#', RESULT='*'):
#     sent = Sentence()
#     for w in ntc_words:
#         print(len(w))
#         for w2 in w:
#             print(w2.form)
#             print(w2.elem[0])
#         elem = w.elem
#         if elem[0] == PRD:
#             sent.set_prd(elem)
#         elif elem[0] == RESULT:
#             sent.set_args(elem)
#         else:
#             sent.words.append(ConllWord(elem))
#     return sent
#
#
# class Eval(object):
#     __metaclass__ = ABCMeta
#
#     def __init__(self):
#         self.corrects = None
#         self.results_sys = None
#         self.results_gold = None
#
#         self.precision = None
#         self.recall = None
#         self.f1 = None
#
#         self.all_corrects = None
#         self.all_results_sys = None
#         self.all_results_gold = None
#
#         self.all_precision = 0.
#         self.all_recall = 0.
#         self.all_f1 = 0.
#         self.nll = 0.
#
#         self._set_params()
#
#     @abstractmethod
#     def _set_params(self):
#         raise NotImplementedError
#
#     @abstractmethod
#     def update_results(self, y_system, y_gold):
#         raise NotImplementedError
#
#     @abstractmethod
#     def _summarize(self):
#         raise NotImplementedError
#
#     @abstractmethod
#     def show_results(self):
#         raise NotImplementedError
#
#     @staticmethod
#     def _get_case_index(y):
#         case_index = -1
#         if y == GA_ID:
#             case_index = 0
#         elif y == O_ID:
#             case_index = 1
#         elif y == NI_ID:
#             case_index = 2
#         return case_index
#
#     @staticmethod
#     def _get_case_type(w_index, p_index, sample):
#         word = sample.sent[w_index]
#         prd = sample.sent[p_index]
#         if word.chunk_index == prd.chunk_index:
#             return BST
#         elif word.chunk_index == prd.chunk_head or word.chunk_head == prd.chunk_index:
#             return DEP
#         return INTRA_ZERO
#
#     @staticmethod
#     def _get_case_name(case_index):
#         assert -1 < case_index < 3
#         case_name = None
#         if case_index == 0:
#             case_name = 'GA'
#         elif case_index == 1:
#             case_name = 'WO'
#         elif case_index == 2:
#             case_name = 'Ni'
#         return case_name
#
#     @staticmethod
#     def _calc_metrics(corrects, results_sys, results_gold):
#         precision = corrects / results_sys
#         recall = corrects / results_gold
#         f1 = 2 * precision * recall / (precision + recall)
#         return precision, recall, f1
#
#
# class SampleEval(Eval):
#
#     def _set_params(self, n_cases=3, n_case_types=3):
#         shape = (n_cases, n_case_types)
#         self.corrects = np.zeros(shape, dtype='float32')
#         self.results_sys = np.zeros(shape, dtype='float32')
#         self.results_gold = np.zeros(shape, dtype='float32')
#
#         self.precision = np.zeros(shape, dtype='float32')
#         self.recall = np.zeros(shape, dtype='float32')
#         self.f1 = np.zeros(shape, dtype='float32')
#
#         self.all_corrects = np.zeros(n_cases, dtype='float32')
#         self.all_results_sys = np.zeros(n_cases, dtype='float32')
#         self.all_results_gold = np.zeros(n_cases, dtype='float32')
#
#     def _summarize(self):
#         p, r, f = self._calc_metrics(self.corrects, self.results_sys, self.results_gold)
#         self.precision = p
#         self.recall = r
#         self.f1 = f
#
#         self.all_corrects = np.sum(self.corrects[:, 1:])
#         self.all_results_sys = np.sum(self.results_sys[:, 1:])
#         self.all_results_gold = np.sum(self.results_gold[:, 1:])
#
#         p, r, f = self._calc_metrics(self.all_corrects, self.all_results_sys, self.all_results_gold)
#         self.all_precision = p
#         self.all_recall = r
#         self.all_f1 = f
#
#     def update_results(self, y_sys_batch, sample):
#         assert len(y_sys_batch) == len(sample)
#         for prd_i, (y_sys, y_gold) in enumerate(zip(y_sys_batch, sample)):
#             assert len(y_sys) == len(y_gold)
#             prd_index = sample.prd_indices[prd_i]
#             self._add_results_gold(sample, prd_index)
#             self._add_results_sys(sample, prd_index, y_sys)
#             self._add_corrects(sample, prd_index, y_sys, y_gold)
#
#     def update_results_old(self, y_sys_batch, sample):
#         assert len(y_sys_batch) == len(sample.y)
#         for prd_i, (y_sys, y_gold) in enumerate(zip(y_sys_batch, sample.y)):
#             assert len(y_sys) == len(y_gold)
#             prd_index = sample.prd_indices[prd_i]
#             self._add_results_gold(sample, prd_index)
#             self._add_results_sys(sample, prd_index, y_sys)
#             self._add_corrects(sample, prd_index, y_sys, y_gold)
#
#     def _add_results_gold(self, sample, prd_index):
#         arg_types = sample.sent[prd_index].arg_types
#         for case_index, arg_type in enumerate(arg_types):
#             if arg_type == BST or arg_type == DEP or arg_type == INTRA_ZERO:
#                 self.results_gold[case_index][arg_type] += 1
#
#     def _add_results_sys(self, sample, prd_index, label_vec):
#         for word_index, label_id in enumerate(label_vec):
#             case_index = self._get_case_index(label_id)
#             case_type = self._get_case_type(word_index, prd_index, sample)
#             if -1 < case_index:
#                 self.results_sys[case_index][case_type] += 1
#
#     def _add_corrects(self, sample, prd_index, y_sys, y_gold):
#         for word_index, (y_hat, y) in enumerate(zip(y_sys, y_gold)):
#             case_index = self._get_case_index(y_hat)
#             case_type = self._get_case_type(word_index, prd_index, sample)
#             if case_index < 0:
#                 continue
#             if y_hat == y:
#                 self.corrects[case_index][case_type] += 1
#
#     def show_results(self):
#         self._summarize()
#         say('\n\tNLL: %f' % self.nll)
#         say('\n\n\tACCURACY')
#
#         for case_index, (crr_c, r_sys_c, r_gold_c) in enumerate(zip(self.corrects,
#                                                                     self.results_sys,
#                                                                     self.results_gold)):
#             ttl_crr = np.sum(crr_c[1:])
#             ttl_res_sys = np.sum(r_sys_c[1:])
#             ttl_res_gold = np.sum(r_gold_c[1:])
#             precision, recall, f1 = self._calc_metrics(ttl_crr, ttl_res_sys, ttl_res_gold)
#
#             case_name = self._get_case_name(case_index)
#             say('\n\tCASE-%s:\n' % case_name)
#             say('\tALL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#                 f1, precision, int(ttl_crr), int(ttl_res_sys), recall, int(ttl_crr), int(ttl_res_gold)))
#
#             for case_type, (crr, r_sys, r_gold) in enumerate(zip(crr_c, r_sys_c, r_gold_c)):
#                 case_type_name = None
#                 if case_type == 1:
#                     case_type_name = 'DEP'
#                 elif case_type == 2:
#                     case_type_name = 'ZERO'
#
#                 if case_type_name is None:
#                     continue
#
#                 say('\t{}:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#                     case_type_name, self.f1[case_index][case_type],
#                     self.precision[case_index][case_type], int(crr), int(r_sys),
#                     self.recall[case_index][case_type], int(crr), int(r_gold)))
#
#         say('\n\tTOTAL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
#             self.all_f1, self.all_precision, int(self.all_corrects), int(self.all_results_sys),
#             self.all_recall, int(self.all_corrects), int(self.all_results_gold)))
#
#
# class BatchEval(Eval):
#
#     def _set_params(self, n_cases=3):
#         self.corrects = np.zeros(n_cases, dtype='float32')
#         self.results_sys = np.zeros(n_cases, dtype='float32')
#         self.results_gold = np.zeros(n_cases, dtype='float32')
#
#         self.precision = np.zeros(n_cases, dtype='float32')
#         self.recall = np.zeros(n_cases, dtype='float32')
#         self.f1 = np.zeros(n_cases, dtype='float32')
#
#         self.all_corrects = 0.
#         self.all_results_sys = 0.
#         self.all_results_gold = 0.
#
#     def _summarize(self):
#         self.precision = self.corrects / self.results_sys
#         self.recall = self.corrects / self.results_gold
#         self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
#
#         self.all_corrects = np.sum(self.corrects)
#         self.all_results_sys = np.sum(self.results_sys)
#         self.all_results_gold = np.sum(self.results_gold)
#
#         self.all_precision = self.all_corrects / self.all_results_sys
#         self.all_recall = self.all_corrects / self.all_results_gold
#         self.all_f1 = 2 * self.all_precision * self.all_recall / (self.all_precision + self.all_recall)
#
#     def update_results(self, batch_y_hat, batch_y):
#         assert len(batch_y_hat) == len(batch_y)
#         assert len(batch_y_hat[0]) == len(batch_y[0]), '%s\n%s' % (str(batch_y_hat), str(batch_y))
#
#         for sent_y_hat, sent_y in zip(batch_y_hat, batch_y):
#             for y_hat, y in zip(sent_y_hat, sent_y):
#                 case_y_hat = self._get_case_index(y_hat)
#                 case_y = self._get_case_index(y)
#
#                 if -1 < case_y_hat:
#                     self.results_sys[case_y_hat] += 1
#                     if y_hat == y:
#                         self.corrects[case_y_hat] += 1
#                 if -1 < case_y:
#                     self.results_gold[case_y] += 1
#
#     def show_results(self):
#         self._summarize()
#         say('\n\tNLL: %f' % self.nll)
#         say('\n\n\tACCURACY')
#
#         for case_index in xrange(3):
#             case_name = self._get_case_name(case_index)
#             if case_name is None:
#                 continue
#
#             crr = int(self.corrects[case_index])
#             r_sys = int(self.results_sys[case_index])
#             r_gold = int(self.results_gold[case_index])
#
#             say('\n\t{}:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
#                 case_name, self.f1[case_index], self.precision[case_index], crr, r_sys,
#                 self.recall[case_index], crr, r_gold))
#
#         crr = int(self.all_corrects)
#         r_sys = int(self.all_results_sys)
#         r_gold = int(self.all_results_gold)
#
#         say('\n\tTOTAL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
#             self.all_f1, self.all_precision, crr, r_sys, self.all_recall, crr, r_gold))
#
#
# class ResultEval(Eval):
#
#     def _set_params(self, n_cases=3, n_case_types=3):
#         shape = (n_cases, n_case_types)
#         self.corrects = np.zeros(shape, dtype='float32')
#         self.results_sys = np.zeros(shape, dtype='float32')
#         self.results_gold = np.zeros(shape, dtype='float32')
#
#         self.precision = np.zeros(shape, dtype='float32')
#         self.recall = np.zeros(shape, dtype='float32')
#         self.f1 = np.zeros(shape, dtype='float32')
#
#         self.all_corrects = np.zeros(n_cases, dtype='float32')
#         self.all_results_sys = np.zeros(n_cases, dtype='float32')
#         self.all_results_gold = np.zeros(n_cases, dtype='float32')
#
#     def _summarize(self):
#         p, r, f = self._calc_metrics(self.corrects, self.results_sys, self.results_gold)
#         self.precision = p
#         self.recall = r
#         self.f1 = f
#
#         self.all_corrects = np.sum(self.corrects[:, 1:])
#         self.all_results_sys = np.sum(self.results_sys[:, 1:])
#         self.all_results_gold = np.sum(self.results_gold[:, 1:])
#
#         p, r, f = self._calc_metrics(self.all_corrects, self.all_results_sys, self.all_results_gold)
#         self.all_precision = p
#         self.all_recall = r
#         self.all_f1 = f
#
#     def update_results(self, batch_y_hat, batch_y):
#         pass
#
#     def calc_results(self, corpus):
#         self._set_params()
#
#         for sent in corpus:
#             for pas in sent.pas:
#                 self._add_results_gold(pas)
#                 self._add_results_sys(pas)
#                 self._add_corrects(pas)
#         self._summarize()
#         self.show_results()
#
#     def _add_results_gold(self, pas):
#         for i, args in enumerate(pas.args_gold):
#             for arg in args:
#                 c = arg.case_type
#                 if c == BST or c == DEP or c == INTRA_ZERO:
#                     self.results_gold[i][c] += 1
#
#     def _add_results_sys(self, pas):
#         for i, args in enumerate(pas.args_sys):
#             for arg in args:
#                 c = arg.case_type
#                 if c == BST or c == DEP or c == INTRA_ZERO:
#                     self.results_sys[i][c] += 1
#
#     def _add_corrects(self, pas):
#         for case_index, (args_sys, args_gold) in enumerate(zip(pas.args_sys, pas.args_gold)):
#             for arg_s in args_sys:
#                 for arg_g in args_gold:
#                     if arg_s.word_index == arg_g.word_index:
#                         self.corrects[case_index][arg_s.case_type] += 1
#                         break
#
#     def show_results(self):
#         self._summarize()
#         say('\n\n\tACCURACY')
#
#         for case_index, (crr_c, r_sys_c, r_gold_c) in enumerate(zip(self.corrects,
#                                                                     self.results_sys,
#                                                                     self.results_gold)):
#             ttl_crr = np.sum(crr_c[1:])
#             ttl_res_sys = np.sum(r_sys_c[1:])
#             ttl_res_gold = np.sum(r_gold_c[1:])
#             precision, recall, f1 = self._calc_metrics(ttl_crr, ttl_res_sys, ttl_res_gold)
#
#             case_name = self._get_case_name(case_index)
#             say('\n\tCASE-%s:\n' % case_name)
#             say('\tALL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#                 f1, precision, int(ttl_crr), int(ttl_res_sys), recall, int(ttl_crr), int(ttl_res_gold)))
#
#             for case_type, (crr, r_sys, r_gold) in enumerate(zip(crr_c, r_sys_c, r_gold_c)):
#                 case_type_name = None
#                 if case_type == 1:
#                     case_type_name = 'DEP'
#                 elif case_type == 2:
#                     case_type_name = 'ZERO'
#
#                 if case_type_name is None:
#                     continue
#
#                 say('\t{}:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#                     case_type_name, self.f1[case_index][case_type],
#                     self.precision[case_index][case_type], int(crr), int(r_sys),
#                     self.recall[case_index][case_type], int(crr), int(r_gold)))
#
#         say('\n\tTOTAL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#             self.all_f1, self.all_precision, int(self.all_corrects), int(self.all_results_sys),
#             self.all_recall, int(self.all_corrects), int(self.all_results_gold)))
#
#
# class PrdEval(Eval):
#
#     def _set_params(self, n_cases=3, n_case_types=3):
#         shape = (6, n_cases, n_case_types)
#         self.corrects = np.zeros(shape, dtype='float32')
#         self.results_sys = np.zeros(shape, dtype='float32')
#         self.results_gold = np.zeros(shape, dtype='float32')
#
#         self.precision = np.zeros(shape, dtype='float32')
#         self.recall = np.zeros(shape, dtype='float32')
#         self.f1 = np.zeros(shape, dtype='float32')
#
#     def _summarize(self):
#         p, r, f = self._calc_metrics(self.corrects, self.results_sys, self.results_gold)
#         self.precision = p
#         self.recall = r
#         self.f1 = f
#
#     def update_results(self, y_sys_batch, sample):
#         assert len(y_sys_batch) == len(sample.y)
#         n_prds = self._bin_prds(len(sample.y))
#         for prd_i, (y_sys, y_gold) in enumerate(zip(y_sys_batch, sample.y)):
#             assert len(y_sys) == len(y_gold)
#             prd_index = sample.prd_indices[prd_i]
#             self._add_results_gold(sample, prd_index, n_prds)
#             self._add_results_sys(sample, prd_index, n_prds, y_sys)
#             self._add_corrects(sample, prd_index, n_prds, y_sys, y_gold)
#
#     @staticmethod
#     def _bin_prds(n_prds):
#         if n_prds < 6:
#             return n_prds - 1
#         return 5
#
#     def _add_results_gold(self, sample, prd_index, n_prds):
#         arg_types = sample.sent[prd_index].arg_types
#         for case_index, arg_type in enumerate(arg_types):
#             if arg_type == BST or arg_type == DEP or arg_type == INTRA_ZERO:
#                 self.results_gold[n_prds][case_index][arg_type] += 1
#
#     def _add_results_sys(self, sample, prd_index, n_prds, label_vec):
#         for word_index, label_id in enumerate(label_vec):
#             case_index = self._get_case_index(label_id)
#             case_type = self._get_case_type(word_index, prd_index, sample)
#             if -1 < case_index:
#                 self.results_sys[n_prds][case_index][case_type] += 1
#
#     def _add_corrects(self, sample, prd_index, n_prds, y_sys, y_gold):
#         for word_index, (y_hat, y) in enumerate(zip(y_sys, y_gold)):
#             case_index = self._get_case_index(y_hat)
#             case_type = self._get_case_type(word_index, prd_index, sample)
#             if case_index < 0:
#                 continue
#             if y_hat == y:
#                 self.corrects[n_prds][case_index][case_type] += 1
#
#     def show_results(self):
#         self._summarize()
#         say('\n\tNLL: %f' % self.nll)
#         say('\n\n\tACCURACY')
#         for n_prds, (crr, res_sys, res_gold) in enumerate(zip(self.corrects, self.results_sys, self.results_gold)):
#             print('\n\tPRD-{}'.format(n_prds))
#             self._show_each_prd_result(crr, res_sys, res_gold, n_prds)
#
#     def _show_each_prd_result(self, corrects, results_sys, results_gold, n_prds):
#         for case_index, (crr_c, r_sys_c, r_gold_c) in enumerate(zip(corrects,
#                                                                     results_sys,
#                                                                     results_gold)):
#             ttl_crr = np.sum(crr_c[1:])
#             ttl_res_sys = np.sum(r_sys_c[1:])
#             ttl_res_gold = np.sum(r_gold_c[1:])
#             precision, recall, f1 = self._calc_metrics(ttl_crr, ttl_res_sys, ttl_res_gold)
#
#             case_name = self._get_case_name(case_index)
#             say('\n\tCASE-%s:\n' % case_name)
#             say('\tALL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#                 f1, precision, int(ttl_crr), int(ttl_res_sys), recall, int(ttl_crr), int(ttl_res_gold)))
#
#             for case_type, (crr, r_sys, r_gold) in enumerate(zip(crr_c, r_sys_c, r_gold_c)):
#                 case_type_name = None
#                 if case_type == 1:
#                     case_type_name = 'DEP'
#                 elif case_type == 2:
#                     case_type_name = 'ZERO'
#
#                 if case_type_name is None:
#                     continue
#
#                 say('\t{}:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#                     case_type_name, self.f1[n_prds][case_index][case_type],
#                     self.precision[n_prds][case_index][case_type], int(crr), int(r_sys),
#                     self.recall[n_prds][case_index][case_type], int(crr), int(r_gold)))
#
#         for index in xrange(2):
#             ttl_crr = np.sum(corrects[:, index+1])
#             ttl_res_sys = np.sum(results_sys[:, index+1])
#             ttl_res_gold = np.sum(results_gold[:, index+1])
#             precision, recall, f1 = self._calc_metrics(ttl_crr, ttl_res_sys, ttl_res_gold)
#
#             say('\tTYPE-{} ALL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#                 index, f1, precision, int(ttl_crr), int(ttl_res_sys), recall, int(ttl_crr), int(ttl_res_gold)))
#
#         ttl_crr = np.sum(corrects[:, 1:])
#         ttl_res_sys = np.sum(results_sys[:, 1:])
#         ttl_res_gold = np.sum(results_gold[:, 1:])
#         precision, recall, f1 = self._calc_metrics(ttl_crr, ttl_res_sys, ttl_res_gold)
#         say('\tTOTAL ALL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
#             f1, precision, int(ttl_crr), int(ttl_res_sys), recall, int(ttl_crr), int(ttl_res_gold)))
