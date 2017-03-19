有n个正整数（每个数小于10亿），将它们表示成字符串形式。对每一个字符串s，可以翻转为新字符串s'，如“1234”可以翻转成“4321”。现在，将这n个字符串以任意顺序连成一个字符环，每个字符串可以选择是否翻转。
在字符环中，从任意一个位置开始，遍历整个环，得到一个长整数。请问，如何才能得到最大的长整数。
from functools import reduce
import numpy as np
class StringCircle():

    #find the max element with two elements
    def __cmp__(self, other,other2):
        min_length = min(len(other),len(other2))
        i = 0
        while i < min_length:
            if other[i] < other2[i] :
                return other2
            elif other[i] > other2[i]:
                return other
            else:
                i += 1
        if i == min_length:
            if len(other) > len(other2):
                return other
            else:
                return other2


    def max_sequence(self,number):
        '''
        :param number:String
        :return: max sequence
        '''
        number_str = str(number)
        number_str_inverse = number_str[::-1]
        sequence_list = []
        index = 0
        length = len(number_str)
        while index < length:
            sequence_list.append(number_str[index:])
            sequence_list.append(number_str_inverse[index:])
            index += 1

        max_num = reduce(lambda x,y:self.__cmp__(x,y),sequence_list)

        j=0
        while j<len(sequence_list):
            if sequence_list[j] == max_num:
                max_index = j
                break
            j += 1

        max_index = np.argmax(sequence_list)
        firstElement = None
        lastElement = None
        if max_index % 2 == 0:
            max_index = max_index // 2
            firstElement = number_str[index:]
            lastElement = number_str[:index]
        else:
            max_index = max_index // 2
            firstElement = number_str_inverse[max_index:]
            lastElement = number_str_inverse[:max_index]
        return [firstElement,lastElement]

    def maxNumberForSelf(self,number):
        number_str = str(number)
        number_str_invese = number_str[::-1]
        if (number_str > number_str_invese) - (number_str < number_str_invese):
            return  number_str_invese
        else:
            return number_str

    def solve(self,number_list):

        numSeqlist = []
        for i in range(len(number_list)):
            numSeqlist.append(self.max_sequence(number_list[i]))

        maxNumberindex = np.argmax(numSeqlist[:][0])

        #maxNumberindex = np.argmax([self.max_sequence(v for v in number_list)],axis=0)

        MaxNumber = self.max_sequence(number_list[maxNumberindex])

        firstElement = MaxNumber[0]
        lastElement = MaxNumber[1]
        middleList = [firstElement]
        for index,v in enumerate(number_list):
            if index != maxNumberindex:
                numberSelf = self.maxNumberForSelf(v)
                index_j = 0
                while index_j < len(middleList) and numberSelf <= middleList[index_j]:
                    index_j += 1
                middleList.insert(index_j,numberSelf)
        middleList.append(lastElement)
        return "".join(middleList)

if __name__ == '__main__':
    s = StringCircle()
    number_list = [892,11,11]
    print(s.solve(number_list))











