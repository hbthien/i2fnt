#!/usr/bin/python3

#This code is built by the comments of @cshallue, with modification in var_to_rename

import sys, getopt
import tensorflow as tf

def main(argv):
   OLD_CHECKPOINT_FILE = "./pretrained_model/model.ckpt-2000000"
   NEW_CHECKPOINT_FILE = "./pretrained_model/model2.ckpt-2000000"
   
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('convert_model.py -i <old checkpoint file> -o <new checkpoint file>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('convert_model.py -i <old checkpoint file> -o <new checkpoint file>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         OLD_CHECKPOINT_FILE = arg
      elif opt in ("-o", "--ofile"):
         NEW_CHECKPOINT_FILE = arg
   print ('Old check point file is: ', OLD_CHECKPOINT_FILE)
   print ('New check point file is: ', NEW_CHECKPOINT_FILE)


   #vars_to_rename = {
   #  "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/weights",
   #  "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/biases",
   #}

   vars_to_rename = {
      "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
      "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
   }

   new_checkpoint_vars = {}
   reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
   for old_name in reader.get_variable_to_shape_map():
      if old_name in vars_to_rename:
         new_name = vars_to_rename[old_name]
      else:
         new_name = old_name
      new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

   init = tf.global_variables_initializer()
   saver = tf.train.Saver(new_checkpoint_vars)

   with tf.Session() as sess:
      sess.run(init)
      saver.save(sess, NEW_CHECKPOINT_FILE)

if __name__ == "__main__":
   main(sys.argv)

