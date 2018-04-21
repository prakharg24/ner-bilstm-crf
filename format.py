import sys
import os
import pdb

# takes 2 arguments: unlabelled file and labelled file, in that order

wo_lbl = open(sys.argv[1],"r")
w_lbl = open(sys.argv[2],"r")

wo_lbl_lines = wo_lbl.readlines()
w_lbl_lines = w_lbl.readlines()

if len(wo_lbl_lines)!=len(w_lbl_lines):
	print ("damn!!! number of lines incorrect... (error!!!)")
	exit(1)

for i in range(len(wo_lbl_lines)):
	wo_items = wo_lbl_lines[i].strip().split(" ")
	w_items = w_lbl_lines[i].strip().split(" ")

	if wo_items[0]=="" and w_items[0]=="":
		continue

	if wo_items[0]=="" and w_items[0]!="":
		print ("no way!!! line %d in labelled file should have been blank... (error!!!)" %(i+1))
		exit(1)

	if len(wo_items)!=1:
		print ("oh no!!! unlabelled file doesn't have exactly 1 token in line %d... (error!!!)" %(i+1))
		exit(1)

	if len(w_items)!=2:
		print ("oh no!!! labelled file doesn't have exactly 2 tokens in line %d... (error!!!)" %(i+1))
		exit(1)

	if w_items[0]!=wo_items[0]:
		print ("woops!!! token mismatch, between labelled and unlabelled files, on line %d... (error!!!)" %(i+1))
		exit(1)

	if w_items[1]!="O" and w_items[1]!="D" and w_items[1]!="T":
		print ("darn!!! label token is different from O,D,T on line %d... (error!!!)" %(i+1))
		exit(1)


print ("congratulations!! your format is spot on!!!")

wo_lbl.close()
w_lbl.close()