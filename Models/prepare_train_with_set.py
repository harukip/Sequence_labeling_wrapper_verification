#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# # Train file create

# Open the DCADE output TableA.txt and read page file to give each node data corresponding labels (column number).

# In[2]:


def train_file_generate(set_total, current_path):
    table_name = os.path.join(current_path, "data", "TableA.txt")
    print("Table Opening:" + table_name + "\n")
    table_a = open(table_name, "r")
    output_name = os.path.join(current_path, "data", "train_raw.txt")
    print("Generating:" + output_name + "\n")

    Set_index = {}

    output = open(output_name, "w")
    output.write("Leafnode\tPTypeSet\tTypeSet\tContentid\tPathid\tSimseqid\tPath\tContent\tLabel\n")

    line = table_a.readline()
    slot = line.rstrip("\n").split("\t") # Read Table line in line and split by \t saved in slot.

    while(slot[0]!="ColType"):
        '''
        Loop until we find the "ColType" in the table file to start generate training data.
        '''
        line = table_a.readline()
        slot = line.rstrip("\n").split("\t")
    print("find coltype")
    col = []
    line = table_a.readline() # Read first line
    while(line != ""):
        slot = line.rstrip("\n").split("\t")
        page_name = slot[0]
        node_col = slot[1:] # Save each column's data in node_col
        col = []
        for i in node_col: # Split Same column data and save in list col
            tmp = i.split(" ")
            col.append(tmp)

        current_page = 0
        for pos in range(len(col)): # Loop all col num
            if len(col[pos]) == 1 and not col[pos][0].isnumeric(): # Find "X-X" like sets
                #print("Find : "+col[pos][0])
                current_page = col[pos][0].split("-")[1]
                current_page = int(current_page)
                set_num = col[pos][0].split("-")[0]
                set_num = int(set_num)
                if set_num not in Set_index.keys():
                    #print("Add new set in dict col num:"+str(pos+1))
                    Set_index[set_num] = pos+1 # Record it in dict

        page_file = pd.read_csv(os.path.join(current_path, "Output", page_name), sep='\t') # Read page file from DECADE Output file.

        col_name = []
        for i in page_file.columns: # Save all column name
            col_name.append(i)

        #print("open "+ page_name)
        label_dict = {} # Define Dict of labels for each node
        for leafnode in page_file[col_name[0]]: # Loop each node in page
            for pos in range(len(col)): # Loop all col
                if str(leafnode) in col[pos]: # if node is in this column
                    if leafnode not in label_dict.keys(): # and if this node not in dict save it
                        label_dict[leafnode] = pos+1 # Shift one column for every node (zero is for empty).
                    #print(str(leafnode) + "\'s label: " + str(pos))
                    break

        for set_count in range(set_total): # Loop every set and give each node a label, which is the column number of the set.
            with open(os.path.join(current_path, "data", "Set-"+str(set_count+1)+".txt"), "r") as set_file:
                line = set_file.readline()
                slot = line.rstrip("\n").split("\t")

                while(slot[0] != "ColType"):
                    line = set_file.readline()
                    slot = line.rstrip("\n").split("\t")

                line = set_file.readline()
                slot = line.rstrip("\n").split("\t")
                while(line!=""):
                    file_set_info = slot[0].split("-")
                    file_set_num = set_count
                    file_set_page = int(file_set_info[1])
                    for node in slot[1:]:
                        if not node.isnumeric():
                            continue
                        node = int(node)
                        if file_set_page == current_page:
                            label_dict[node] = Set_index[set_count+1]
                    line = set_file.readline()
                    slot = line.rstrip("\n").split("\t")
        error = False
        '''for i in range(max(label_dict.keys())+1):
            if i not in label_dict:
                error = True
                print(i)'''
        #print(label_dict)
        #count = 0
        data_list = [[], [], [], [], [], [], [], [], []]
        for i in range(max(label_dict.keys())+1): # Loop each recorded node and output as train file
            Leafnode = page_file[col_name[0]][i]
            PTypeSetid = page_file[col_name[6]][i]
            TypeSetid = page_file[col_name[7]][i]
            Contentid = page_file[col_name[8]][i].split("-")[1]
            Pathid = page_file[col_name[9]][i]
            SimSeqid = page_file[col_name[10]][i]
            Path = page_file[col_name[2]][i]
            Content = page_file[col_name[1]][i]
            cols = [Leafnode, PTypeSetid, TypeSetid, Contentid, Pathid, SimSeqid, Path, Content]
            for c in range(len(cols)):
                output.write(str(cols[c]) + "\t")
                data_list[c].append(cols[c])
            if i not in label_dict:
                output.write("0\n")
                data_list[len(cols)].append(0)
            else:
                output.write(str(label_dict[i]) + "\n")
                data_list[len(cols)].append(label_dict[i])
        line = table_a.readline()
    output.close()
    data = pd.DataFrame(np.transpose(np.array(data_list)), 
                 columns=["Leafnode", "PTypeSet", "TypeSet", "Contentid", "Pathid", "Simseqid", "Path", "Content", "Label"])
    return data, Set_index


# In[3]:


if __name__ == "__main__":
    set_total = 0
    current_path = os.path.join(os.path.expanduser("~"), "jupyter", "web_verification")
    print(current_path)
    data, Set_index = train_file_generate(set_total, current_path)
    if set_total > 0:
        with open(os.path.join(current_path, "data", "Set_idx.txt"), "w") as set_file:
            set_file.write(str(Set_index))
        print(Set_index)


# In[ ]:




