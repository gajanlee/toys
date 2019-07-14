from model import *
from data import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print = logger.warning

teacher_forcing_ratio = 1.0
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Train Trick
# 1. teacher forcing
#   If teacher, we will let the target as next time step input
#   Else, next input is previous output.
#
# 2. gradient clipping

def train(input_variable, lengths, target_variable, mask, max_target_len, 
            encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, 
            batch_size, clip, max_length=MAX_LENGTH):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable#.to(device)
    lengths = lengths#.to(device)
    target_variable = target_variable#.to(device)
    mask = mask#.to(device)

    # Initialize vaariables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Initialize decoder input
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input#.to(device)

    # Initalize decoder hidden state with the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1)
            
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Use the decoder output as next step input
            _, topi = decoder_output#.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input#.to(device)
            
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()
    _ = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, 
                encoder, decoder, encoder_optimizer, decoder_optimizer, 
                embedding, encoder_n_layers, decoder_n_layers, 
                save_dir, n_iteration, batch_size, print_every, save_every, 
                clip, loadFilename=None):
    
    training_batches = [voc.batch2TrainData([random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    #print(voc.batch2TrainData([pairs[0]]))
    #print(pairs[0])
    #print(voc.word2index)
    #exit()
    # Initializations  
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, 
                        encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
                        batch_size, clip)
        print_loss += loss
        
        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

            for eval_word in ["abie", "mike", "root", "taylor", "brook", "brooke"]:
                print(f"{eval_word} => {evaluation(encoder, decoder, voc, eval_word)}")
            print("=============================")
        
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, "nameTrans", '{}-{}'.format(encoder_n_layers, decoder_n_layers))#, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
        

def evaluate(encoder, decoder, searcher, voc, word, max_length):
    # batch_size is 1
    indexes_batch = [voc.indexesFromWords(word)]
    print(indexes_batch)
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    input_batch = input_batch#.to(device)
    lengths = lengths#.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def train_model():
    model_name = "nt_model"
    attn_model = "dot"
    save_dir = "save"

    pairs = list(name_pairs_generator())
    
    dim_size = 10
    hidden_size = 30
    encoder_n_layers = 1
    decoder_n_layers = 1
    dropout = 0.1
    batch_size = 32

    loadFilename = None
    checkpoint_iter = 200

    voc = getTranslateNameVocabulary()

    embedding = nn.Embedding(voc.num_words, hidden_size)
    
    encoder = TranslatorEncoder(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = TranslatorDecoder(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder = encoder#.to(device)
    decoder = decoder#.to(device)

    clip = 50.0 
    learning_rate = 0.0001
    decoder_learning_rate = 5.0
    n_iteration = 40000
    print_every = 10
    save_every = n_iteration / 4

    # train mode
    encoder.train()
    decoder.train()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)


    trainIters(model_name, voc, pairs, 
                encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, 
                save_dir, n_iteration, batch_size, print_every, save_every,
                clip)

    return encoder, decoder, voc

def evaluation(encoder, decoder, voc, sentence):
    encoder.eval()
    decoder.eval()
    #sentence = ["SOS"] + list(sentence)[:5] + ["EOS"] + ["PAD"]*7
    #sentence = sentence[:7]

    # Initializer search module
    searcher = GreedySearchDecoder(encoder, decoder)
    
    decode_words = evaluate(encoder, decoder, searcher, voc, sentence, 7)
    
    return decode_words

def main():
    encoder, decoder, voc = train_model()

    eval_dataset = [
        "mike",
        "axel",
        "aviva",
        "avi",
        "qiana",
        "quaid",
        "lacy",
        "lael",
        "laddle",
    ]


    for name in eval_dataset:
        print(f"{name} to ", evaluation(encoder, decoder, voc, name))

    for name in eval_dataset:
        print(f"{name} to {evaluation(encoder, decoder, voc, name)}")

if __name__ == "__main__":
    main()
