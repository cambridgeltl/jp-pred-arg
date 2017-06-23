from bp_pas.ling.word import Word, ConllWord
from bp_pas.ling.sent import Sentence
from bp_pas.ling.pas import Predicate, Argument, PAS
from bp_pas.vocab import Vocab, PAD, UNK

class NTCLoader():

    def __init__(self):
        self.BOD = '#'
        self.BOC = '*'
        self.EOS = 'EOS'

    def load_corpus(self, path_to_file, max_sents=-1):
        lines = list(open(path_to_file))
        chunks = self.read_chunks(lines, delim=self.EOS)[:max_sents]
        return [[self.chunk_to_sentence(chunk) for chunk in chunks]]

    def read_chunks(self, lines, delim):
        chunks = []
        chunk = []
        for line in lines:
            if line.startswith(delim) and len(chunk) > 0:
                chunks.append(chunk)
                chunk = []
            else:
                chunk.append(line.strip())
        return chunks

    def chunk_to_sentence(self, chunk):
        words = []
        chunk_index = None
        chunk_head = None
        for line in chunk:
            elem = line.split()
            if line.startswith('#'):
                pass
            elif line.startswith(self.BOC):
                chunk_index, chunk_head = self._get_chunk_info(elem)
            else:
                word = self._get_word(w_index=len(words),
                                  chunk_index=chunk_index,
                                  chunk_head=chunk_head,
                                  sent_index=0, #len(doc),
                                  elem=elem)
                words.append(word)
        for w in words:
            w.set_cases(words)
        return Sentence(words, self.pas_from_words(words))

    def pas_from_words(self, words):
        types = ['GA', 'O', 'NI']
        pases = []
        for word in words:
            if word.is_prd:
                pred = Predicate(word.index, word.form)
                args = []
                for i in range(len(types)):
                    arg_idx = word.arg_indices[i]
                    if arg_idx >= 0:
                        args.append(Argument(words[arg_idx].index,
                                             words[arg_idx].form,
                                             types[i]))
#                        print(args[-1])
                pases.append(PAS(pred, args))
        return pases

    @staticmethod
    def _get_doc_id(elem):
        return elem[1].split(':')[1].split('-')[0]

    @staticmethod
    def _get_chunk_info(elem):
        return int(elem[1]), int(elem[2][:-1])

    @staticmethod
    def _get_word(w_index, chunk_index, chunk_head, sent_index, elem):
        w = Word(w_index, elem)
        w.sent_index = sent_index
        w.chunk_index = chunk_index
        w.chunk_head = chunk_head
        return w

    class CONLLLoader():

        def load_corpus(self, path):
            PRD = '#'
            RESULT = '*'
            corpus = []
            with open(path) as f:
                sent = Sentence()
                for line in f:
                    line = line.rstrip()
                    elem = line.split('\t')

                    if len(line) == 0:
                        corpus.append(sent)
                        sent = Sentence()
                    elif elem[0] == PRD:
                        sent.set_prd(elem)
                    elif elem[0] == RESULT:
                        sent.set_args(elem)
                    else:
                        sent.words.append(ConllWord(elem))
            return corpus












            #
    # def read_next_sent(self, f, line_offset = 0):
    #     prev_doc_id = None
    #     doc = []
    #     chunk_index = None
    #     chunk_head = None
    #     sent = Sentence()
    #     for line in f[line_offset:]:
    #         elem = line.rstrip().split()
    #         #                print('line:-{}-'.format(elem))
    #         if line.startswith(self.BOD):
    #             #                    print('BOD')
    #             doc_id = self._get_doc_id(elem)
    #             if prev_doc_id and prev_doc_id != doc_id:
    #                 prev_doc_id = doc_id
    #                 corpus.append(doc)
    #                 doc = []
    #             elif prev_doc_id is None:
    #                 prev_doc_id = doc_id
    #         elif line.startswith(self.BOC):
    #             #                    print('BOC')
    #             chunk_index, chunk_head = self._get_chunk_info(elem)
    #         elif line.startswith(self.EOS):
    #             #                    print('EOS')
    #             for w in sent:
    #                 w.set_cases(sent)
    #             doc.append(sent)
    #             sent = []
    #         else:
    #             word = self._get_word(w_index=len(sent),
    #                                   chunk_index=chunk_index,
    #                                   chunk_head=chunk_head,
    #                                   sent_index=len(doc),
    #                                   elem=elem)
    #             sent.append(word)
    #         if len(corpus) == max_data_size:
    #             break
    #     else:
    #         if doc:
    #             corpus.append(doc)
    #     return sent
    #





        #         corpus = []
        #         offset = 0
        #         with open(path) as f:
        #             chunks = self.read_file_chunks(f)
        #             print(len(chunks))
        #             for l in chunks[0]:
        #                 print(l)
        # #            print(chunks[0])
        # #            next_sent, next_offset = self.read_next_sent(f, offset)
        # #            corpus.append(next_sent)
        # #            offset = next_offset
        #        return corpus

#     def load_corpus(self, path, max_data_size=-1):
#         if path is None:
#             return None
#
#         BOD = '#'
#         BOC = '*'
#         EOS = 'EOS'
#
#         corpus = []
#         with open(path) as f:
#             prev_doc_id = None
#             doc = []
#             sent = []
#             chunk_index = None
#             chunk_head = None
#             for line in f:
#                 elem = line.rstrip().split()
# #                print('line:-{}-'.format(elem))
#                 if line.startswith(BOD):
# #                    print('BOD')
#                     doc_id = self._get_doc_id(elem)
#                     if prev_doc_id and prev_doc_id != doc_id:
#                         prev_doc_id = doc_id
#                         corpus.append(doc)
#                         doc = []
#                     elif prev_doc_id is None:
#                         prev_doc_id = doc_id
#                 elif line.startswith(BOC):
# #                    print('BOC')
#                     chunk_index, chunk_head = self._get_chunk_info(elem)
#                 elif line.startswith(EOS):
# #                    print('EOS')
#                     for w in sent:
#                         w.set_cases(sent)
#                     doc.append(sent)
#                     sent = []
#                 else:
#                     word = self._get_word(w_index=len(sent),
#                                           chunk_index=chunk_index,
#                                           chunk_head=chunk_head,
#                                           sent_index=len(doc),
#                                           elem=elem)
#                     sent.append(word)
#                 if len(corpus) == max_data_size:
#                     break
#             else:
#                 if doc:
#                     corpus.append(doc)
#         return corpus


