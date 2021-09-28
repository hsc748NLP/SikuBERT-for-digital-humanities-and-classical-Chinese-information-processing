#!/usr/bin/env python
from onmt.bin.translate import main



def translate(input_text):
    '''
    :param input_text: 从浏览器前端文本框输入的原始文本
    :return: processed_text: 处理后可直接用于前端呈现的文本
    '''

    def writetxt(path, text):
        with open(path, 'w', encoding='utf8') as f:
            f.write(str(text))
        f.close()

    def readtxt(path):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = lines[0].replace('\n','')
        return lines

    writetxt('./data/src-test.txt', input_text)
    main()
    processed_text = readtxt('./data/pred.txt')

    return processed_text



if __name__ == "__main__":
    input_text = '虽然 ， 每 至 于 族 ， 吾 见 其 难为 ， 怵然 为戒 ， 视 为 止 ， 行 为迟 。'
    output = translate(input_text)
    print(output)