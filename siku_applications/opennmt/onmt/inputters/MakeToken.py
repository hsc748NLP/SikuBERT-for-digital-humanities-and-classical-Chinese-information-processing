import sentencepiece as spm
import MeCab
import pandas as pd

def korean_token(datatxt):
    m = MeCab.Tagger()
    delete_tag = ['BOS/EOS', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC']

    def del_post_pos(sentence):
        tokens = sentence.split()  # 원본 문장 띄어쓰기로 분리

        dict_list = []

        for token in tokens:  # 띄어쓰기로 분리된 각 토큰 {'단어':'형태소 태그'} 와 같이 딕셔너리 생성
            m.parse('')
            node = m.parseToNode(token)
            word_list = []
            morph_list = []

            while node:
                morphs = node.feature.split(',')
                word_list.append(node.surface)
                morph_list.append(morphs[0])
                node = node.next

            dict_list.append(dict(zip(word_list, morph_list)))

        for dic in dict_list:  # delete_tag에 해당하는 단어 쌍 지우기 (조사에 해당하는 단어 지우기)
            for key in list(dic.keys()):
                if dic[key] in delete_tag:
                    del dic[key]

        combine_word = [''.join(list(dic.keys())) for dic in dict_list]  # 형태소로 분리된 각 단어 합치기
        result = ' '.join(combine_word)  # 띄어쓰기로 분리된 각 토큰 합치기

        return result  # 온전한 문장을 반환

    data = open(datatxt,'r', encoding='utf-8')

    with open("data/kor.txt", "w", encoding='utf-8') as f:
        for row in data:
            f.write(del_post_pos(row))
            f.write('\n')

    spm.SentencePieceTrainer.Train(
        '--input=data/kor.txt \
        --model_prefix=data/korean_tok \
        --vocab_size=100000 \
        --hard_vocab_limit=false'
    )

def english_token(datatxt):
    data = open(datatxt,'r', encoding='utf-8')

    with open("data/eng.txt", "w", encoding='utf-8') as f:
        for row in data:
            f.write(row)
            f.write('\n')

    spm.SentencePieceTrainer.Train(
        '--input=data/eng.txt \
        --model_prefix=data/english_tok \
        --vocab_size=100000 \
        --hard_vocab_limit=false'
    )

