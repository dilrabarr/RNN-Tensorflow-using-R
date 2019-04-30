#Importing necessary library
library(tensorflow)
library(keras)
#Sample text from Obama's tweet 
text = "My heart goes out to the victims and their families in London. 
        No act of terror can shake the strength and resilience of our British ally.Chuck Berry rolled over everyone who came before him – and turned up everyone who came after. 
We'll miss you, Chuck. Be good.On International Women’s Day, @MichelleObama and I are inspired by all of you who embrace your power to drive change.
Humbled to be recognized by a family with a legacy of service. Who's your #ProfileInCourage? Tell me about them:"

text <- unlist(strsplit(text, ' ', fixed=TRUE))
dictionary <- sort(unique(text))
text_v <- rep(0, length(text))
for(i in 1:length(dictionary)){
  text_v[which(text == dictionary[[i]])]<-i
}

# RNN set-up
#parameters
learning_rate<- 0.001
training_iters <- 10000L
display_step <- 1000L
n_input <- 3L
n_hidden <- 512L
n_steps<-10L

#Variables
vocab_size <- length(dictionary)

#TensorFlow input
x <- tf$placeholder(tf$float32, shape(NULL, n_input))
y <- tf$placeholder(tf$float32, shape(NULL, vocab_size))

#Weights and Biases for RNN
weights <- tf$Variable(tf$random_normal(shape(n_hidden, vocab_size)))
biases <- tf$Variable(tf$random_normal(shape(vocab_size)))

RNN<-function(x, weights, biases){
  
  x <- tf$reshape(x, c(-1L, n_input))
  x <- tf$split(x, n_input, 1L)
  
  rnn_cell<-tf$nn$rnn_cell$MultiRNNCell(list(tf$nn$rnn_cell$BasicLSTMCell(n_hidden), tf$nn$rnn_cell$BasicLSTMCell(n_hidden)))
  
  outputs<-tf$nn$static_rnn(cell = rnn_cell, inputs = x, dtype = tf$float32)
  
  return(tf$matmul(outputs[[1]][[3]], weights) + biases)
}

pred<-RNN(x, weights, biases)

cost<-tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer <- tf$train$RMSPropOptimizer(learning_rate=learning_rate)$minimize(cost)

correct_pred <- tf$equal(tf$argmax(pred, 1L), tf$argmax(y, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_pred, tf$float32))
#Train the model
init <- tf$global_variables_initializer()

sess <- tf$Session()
sess$run(init)
#Training Model
step <- 0
batch_start <- sample(1:(length(text_v)-n_input-n_steps), 1)
loss_total <- 0
acc_total <- 0

library(tictoc)
install.packages('tictoc')
tic()

while(step < training_iters){
  batchx<-t(as.matrix(sapply(c(0:(n_steps-1)), function(x) text_v[(batch_start+x):(batch_start+x+n_input-1)])))
  batchy<-matrix(0,10,72)
  hoty<-sapply(c(0:(n_steps-1)), function(x) text_v[(batch_start+x+n_input)])
  for(i in 1:n_steps){
    batchy[i,hoty[i]]<-1
  }
  runs<-sess$run(c(optimizer, accuracy, cost, pred), feed_dict = dict(x = batchx, y=batchy))
  acc<-runs[[2]]
  loss<-runs[[3]]
  p<-runs[[4]]
  
  loss_total <- loss_total+loss
  acc_total <- acc_total+acc
  
  if(step %% display_step == 0){
    message(paste0("Step=", step, ", Avg. Loss=",round(loss_total/display_step, digits=4),
                   ", Avg. Acc.=", round(acc_total/display_step, digits=4),
                   ", [", dictionary[which.max(batchy[n_steps,])],
                   "] vs. [", dictionary[which.max(p[n_steps,])], "]\n"))
    acc_total<-0
    loss_total<-0
  }
  step <- step+1
  batch_start <- sample(1:(length(text_v)-n_input-n_steps), 1)
}
#Play with the model
length_of_sentence<- 10

input_text<-"terror terror terror"
input_text <- unlist(strsplit(input_text, ' ', fixed=TRUE))
input_v <- rep(0, length(input_text))
for(i in 1:length(dictionary)){
  input_v[which(input_text == dictionary[[i]])]<-i
}

sent_v<-input_v

for(i in 1:length_of_sentence){
  batchx<-matrix(input_v, nrow = 1)
  p<-which.max(sess$run(pred, feed_dict = dict(x = batchx)))
  sent_v<-c(sent_v, p)
  input_v<-c(input_v, p)[2:4]
}

sent_w<-rep("", length(sent_v))
for(i in 1:length(sent_w)){
  sent_w[i]<-dictionary[sent_v[i]]
}
paste(sent_w, collapse = " ")

#Output is "[1] "terror terror terror and resilience of our British ally.Chuck Berry rolled over everyone"
