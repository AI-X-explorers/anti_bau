import numpy as np
import copy

hydrophily = ["S","T","C","Y","N","Q","D","E","K","R","H"] # 亲水性
hydrophobe = ["G","A","V","L","I","M","F","W","P"] # 疏水性
e_charge = {"S":0,"T":0,"C":0,"Y":0,"N":0,"Q":0,"D":-1,"E":-1,"K":1,"R":1\
            ,"H":1,"G":0,"A":0,"V":0,"L":0,"I":0,"M":0,"F":0,"W":0,"P":0}

def get_sequence(sequence_now,charge_now,rule,rule_count):
    
    if rule_count==peptide_len:
        if charge_now>0:
            result_list[rules_list_8.index(rule)].append(sequence_now)
            
    else:
        p_type = int(rule[rule_count])
        if p_type == 1:
            for pep in hydrophily:
                get_sequence(sequence_now+pep,charge_now+e_charge[pep],rule,rule_count+1)
        else:
            for pep in hydrophobe:
                get_sequence(sequence_now+pep,charge_now+e_charge[pep],rule,rule_count+1)


if __name__ == "__main__":
    
    # rules_list_7 = ["0000011","0000111",
    #             "0001111","0011111",
    #             "1100000","1110000",
    #             "1111000","1111100",
    #             "1100100","0010011",
    #             "1001001","1001000",
    #             "1000100","0100100",
    #             "0010010","0010001",
    #             "0001001"]

    rules_list_8 = ['11001000', '01001000', '10010001', '10010000', 
                    '11000100', '01000100', '10001001', '10001000', 
                    '10010001', '00010001', '00100011', '00100010', 
                    '10001001', '00010011', '00010010', '00001001', 
                    '11110000', '11100000', '11111000', '00001111', 
                    '00000111', '00011111', '11001001', '01001001', 
                    '10010011', '10010010', '00100100', '00100101', 
                    '10100100']


    peptide_len = 8

    result_list = [[] for x in rules_list_8]


    count = 0
    for rule in rules_list_8:
        get_sequence("",0,rule,0)
        count+=1
        print(count)
    file_count = 0
    # all_peptides = []
    peptides_num = 0
    for result in result_list:
        f = open("./antibact_final_training/8_peptide/8_peptide_rule_%d.txt"%(file_count),"w")
        for r in result:
            f.write(r+"\n")
        f.close()
        file_count+=1
        peptides_num += len(result)
        # all_peptides.extend(result)
    print("total peptides: {}".format(peptides_num))

    # f_name = './antibact_final_training/Hexapeptides_new.txt'
    # with open(f_name,'w') as f:
    #     for r in all_peptides:
    #         f.write(r+"\n")

    # merge new peptides with previous Hexapeptides:
    # ori_file = 'antibact_final_training/Hexapeptides.txt'
    # files = [f_name,ori_file]
    # obj_file = 'antibact_final_training/Hexapeptides_merged.txt'
    # with open(obj_file,'w') as f:
    #     for file in files:
    #         for line in open(file):
    #             f.writelines(line)