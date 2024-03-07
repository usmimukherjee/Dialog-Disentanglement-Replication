#!/usr/bin/env python

import argparse
import random
import sys
import string
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reserved_words import reserved

global parser, args, WEIGHT_DECAY, HIDDEN, LEARNING_RATE, LEARNING_DECAY_RATE, MOMENTUM, EPOCHS, DROP, MAX_DIST, OUTPUT, FEATURES, common_short_names, cache
cache = {}
common_short_names = {"ng", "_2", "x_", "rq", "\\9", "ww", "nn", "bc", "te", 
                          "io", "v7", "dm", "m0", "d1", "mr", "x3", "nm", "nu", "jc", "wy", "pa", "mn",
                          "a_", "xz", "qr", "s1", "jo", "sw", "em", "jn", "cj", "j_"}

def initializing():
    ''' This method to set up command-line argument parsing. 
    It defines various parameters for training or testing that can be 
    passed to a script from the command line, '''
    # General arguments
    parser.add_argument('prefix', help="Start of names for files produced.")

    # Data arguments
    parser.add_argument('--train', nargs="+", help="Training files, e.g. train/*annotation.txt")
    parser.add_argument('--dev', nargs="+", help="Development files, e.g. dev/*annotation.txt")
    parser.add_argument('--test', nargs="+", help="Test files, e.g. test/*annotation.txt")
    parser.add_argument('--test-start', type=int, help="The line to start making predictions from in each test file.", default=1000)
    parser.add_argument('--test-end', type=int, help="The line to stop making predictions on in each test files.", default=1000000)
    parser.add_argument('--model', help="A file containing a trained model")
    parser.add_argument('--random-sample', help="Train on only a random sample of the data with this many examples.")

    # Model arguments
    parser.add_argument('--hidden', default=512, type=int, help="Number of dimensions in hidden vectors.")
    parser.add_argument('--word-vectors', help="File containing word embeddings.")
    parser.add_argument('--layers', default=2, type=int, help="Number of hidden layers in the model")
    parser.add_argument('--nonlin', choices=["tanh", "cube", "logistic", "relu", "elu", "selu", "softsign", "swish", "linear"], default='softsign', help="Non-linearity type.")

    # Inference arguments
    parser.add_argument('--max-dist', default=101, type=int, help="Maximum number of messages to consider when forming a link (count includes the current message).")
    parser.add_argument('--dynet-autobatch', action='store_true', help="Use dynet autobatching.")
    parser.add_argument('--no_autobatch', action='store_true', help='Disable autobatching')


    # Training arguments
    parser.add_argument('--report-freq', default=5000, type=int, help="How frequently to evaluate on the development set.")
    parser.add_argument('--epochs', default=20, type=int, help="Maximum number of epochs.")
    parser.add_argument('--opt', choices=['sgd', 'mom'], default='sgd', help="Optimisation method.")
    parser.add_argument('--seed', default=10, type=int, help="Random seed.")
    parser.add_argument('--weight-decay', default=1e-7, type=float, help="Apply weight decay.")
    parser.add_argument('--learning-rate', default=0.018804, type=float, help="The initial learning rate.")
    parser.add_argument('--learning-decay-rate', default=0.103, type=float, help="The rate at which the learning rate decays.")
    parser.add_argument('--momentum', default=0.1, type=float, help="Hyperparameter for momentum.")
    parser.add_argument('--drop', default=0.0, type=float, help="Dropout, applied to inputs only.")
    parser.add_argument('--clip', default=3.740, type=float, help="Gradient clipping.")

def header(args, out=sys.stdout):
    ''' This method is used to print the header of the output file.'''
    head_text = "# "+ time.ctime(time.time())
    head_text += "\n# "+ ' '.join(args)
    for outfile in out:
        print(head_text, file=outfile)
        
def update_user(users, user):
    '''This function updates the set of users with the provided username 
    if it's not a reserved keyword or composed only of digits.'''
    if user in reserved:
        return
    all_digit = True
    for char in user:
        if char not in string.digits:
            all_digit = False
    if all_digit:
        return
    users.add(user.lower())

def update_users(line, users):
    ''' This method processes a line of text to extract and update user information 
    in the provided users collection. It filters out non-user-related entries and handles
    different formats of user names.'''
    if len(line) < 2:
        return
    user = line[1]
    if user in ["Topic", "Signoff", "Signon", "Total", "#ubuntu"
            "Window", "Server:", "Screen:", "Geometry", "CO,",
            "Current", "Query", "Prompt:", "Second", "Split",
            "Logging", "Logfile", "Notification", "Hold", "Window",
            "Lastlog", "Notify", 'netjoined:']:
        # Ignore as these are channel commands
        pass
    else:
        if line[0].endswith("==="):
            parts = ' '.join(line).split("is now known as")
            if len(parts) == 2 and line[-1] == parts[-1].strip():
                user = line[-1]
        elif line[0][-1] == ']':
            if user[0] == '<':
                user = user[1:]
            if user[-1] == '>':
                user = user[:-1]

        user = user.lower()
        update_user(users, user)
        # This is for cases like a user named |blah| who is
        # refered to as simply blah
        core = [char for char in user]
        while len(core) > 0 and core[0] in string.punctuation:
            core.pop(0)
        while len(core) > 0 and core[-1] in string.punctuation:
            core.pop()
        core = ''.join(core)
        update_user(users, core)

def get_targets(line, users):
    '''This function extracts a set of target usernames 
    from a line of text, ensuring they are actual users 
    and not punctuation or short, irrelevant tokens.'''
    targets = set()
    for token in line[2:]:
        token = token.lower()
        user = None
        if token in users and len(token) > 2:
            user = token
        else:
            core = [char for char in token]
            while len(core) > 0 and core[-1] in string.punctuation:
                core.pop()
                nword = ''.join(core)
                if nword in users and (len(core) > 2 or nword in common_short_names):
                    user = nword
                    break
            if user is None:
                while len(core) > 0 and core[0] in string.punctuation:
                    core.pop(0)
                    nword = ''.join(core)
                    if nword in users and (len(core) > 2 or nword in common_short_names):
                        user = nword
                        break
        if user is not None:
            targets.add(user)
    return targets

def lines_to_info(text_ascii):
    ''' This function processes chat log lines to 
    extract structured information including users, timestamps, and message targets
    It keeps track of user interactions 
    and message sequencing for analysis and feature extraction.'''
    users = set()
    for line in text_ascii:
        update_users(line, users)

    chour = 12
    cmin = 0
    info = []
    target_info = {}
    nexts = {}
    for line_no, line in enumerate(text_ascii):
        if line[0].startswith("["):
            user = line[1][1:-1]
            nexts.setdefault(user, []).append(line_no)

    prev = {}
    for line_no, line in enumerate(text_ascii):
        user = line[1]
        system = True
        if line[0].startswith("["):
            chour = int(line[0][1:3])
            cmin = int(line[0][4:6])
            user = user[1:-1]
            system = False
        is_bot = (user == 'ubottu' or user == 'ubotu')
        targets = get_targets(line, users)
        for target in targets:
            target_info.setdefault((user, target), []).append(line_no)
        last_from_user = prev.get(user, None)
        if not system:
            prev[user] = line_no
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]

        info.append((user, targets, chour, cmin, system, is_bot, last_from_user, line, next_from_user))

    return info, target_info

def get_time_diff(info, a, b):
    '''This function calculates the time difference in minutes between two messages, 
    identified by their indices (a and b) in the
    info list.'''
    if a is None or b is None:
        return -1
    if a > b:
        t = a
        a = b
        b = t
    ahour = info[a][2]
    amin = info[a][3]
    bhour = info[b][2]
    bmin = info[b][3]
    if ahour == bhour:
        return bmin - amin
    if bhour < ahour:
        bhour += 24
    return (60 - amin) + bmin + 60*(bhour - ahour - 1)

def get_features(name, query_no, link_no, text_ascii, text_tok, info, target_info, do_cache):
    '''feature extraction function that takes a name, query_no, link_no, text_ascii,
    text_tok, info, target_info, and do_cache as input and returns a list of features.'''
    # Check if the features for this combination of name, query_no, and link_no are cached,
    # if so return the cached result to avoid recomputation.
    if (name, query_no, link_no) in cache:
        return cache[name, query_no, link_no]

    features = []
    
    # Extract information for the query and the link from the 'info' structure based on their indices.
    # The 'info' structure seems to contain various details about each message, such as user information,
    # target users, timestamp, system message flag, bot flag, and neighboring message details.
    quser, qtargets, qhour, qmin, qsystem, qis_bot, qlast_from_user, qline, qnext_from_user = info[query_no]
    luser, ltargets, lhour, lmin, lsystem, lis_bot, llast_from_user, lline, lnext_from_user = info[link_no]

    # General information about this sample of data
    # Year
    for i in range(2004, 2018):
        features.append(str(i) in name)
    # Number of messages per minute
    start = None
    end = None
    for i in range(len(text_ascii)):
        if start is None and text_ascii[i][0].startswith("["):
            start = i
        if end is None and i > 0 and text_ascii[-i][0].startswith("["):
            end = len(text_ascii) - i - 1
        if start is not None and end is not None:
            break
    diff = get_time_diff(info, start, end)
    msg_per_min = len(text_ascii) / max(1, diff)
    cutoffs = [-1, 1, 3, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= msg_per_min < end)

    # Query
    #  - Normal message or system message
    features.append(qsystem)
    #  - Hour of day
    features.append(qhour / 24)
    #  - Is it targeted
    features.append(len(qtargets) > 0)
    #  - Is there a previous message from this user?
    features.append(qlast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if qlast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[qlast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if qlast_from_user is None else query_no - qlast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, query_no, qlast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(qis_bot)

    # Link
    #  - Normal message or system message
    features.append(lsystem)
    #  - Hour of day
    features.append(lhour / 24)
    #  - Is it targeted
    features.append(link_no != query_no and len(ltargets) > 0)
    #  - Is there a previous message from this user?
    features.append(link_no != query_no and llast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if link_no == query_no or llast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[llast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if llast_from_user is None else link_no - llast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(link_no != query_no and start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, link_no, llast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(lis_bot)
    #  - Is the message after from the same user?
    features.append(link_no != query_no and link_no + 1 < len(info) and luser == info[link_no + 1][0])
    #  - Is the message before from the same user?
    features.append(link_no != query_no and link_no - 1 > 0 and luser == info[link_no - 1][0])

    # Both
    #  - Is this a self-link?
    features.append(link_no == query_no)
    #  - How far apart in messages are the two?
    dist = query_no - link_no
    features.append(min(100, dist) / 100)
    features.append(dist > 1)
    #  - How far apart in time are the two?
    time = get_time_diff(info, link_no, query_no)
    features.append(min(100, time) / 100)
    cutoffs = [-1, 0, 1, 5, 60, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Does the link target the query user?
    features.append(quser.lower() in ltargets)
    #  - Does the query target the link user?
    features.append(luser.lower() in qtargets)
    #  - none in between from src?
    features.append(link_no != query_no and (qlast_from_user is None or qlast_from_user < link_no))
    #  - none in between from target?
    features.append(link_no != query_no and (lnext_from_user is None or lnext_from_user > query_no))
    #  - previously src addressed target?
    #  - future src addressed target?
    #  - src addressed target in between?
    if link_no != query_no and (quser, luser) in target_info:
        features.append(min(target_info[quser, luser]) < link_no)
        features.append(max(target_info[quser, luser]) > query_no)
        between = False
        for num in target_info[quser, luser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - previously target addressed src?
    #  - future target addressed src?
    #  - target addressed src in between?
    if link_no != query_no and (luser, quser) in target_info:
        features.append(min(target_info[luser, quser]) < link_no)
        features.append(max(target_info[luser, quser]) > query_no)
        between = False
        for num in target_info[luser, quser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - are they the same speaker?
    features.append(luser == quser)
    #  - do they have the same target?
    features.append(link_no != query_no and len(ltargets.intersection(qtargets)) > 0)
    #  - Do they have words in common?
    ltokens = set(text_ascii[link_no])
    qtokens = set(text_ascii[query_no])
    common = len(ltokens.intersection(qtokens))
    if link_no != query_no and len(ltokens) > 0 and len(qtokens) > 0:
        features.append(common / len(ltokens))
        features.append(common / len(qtokens))
    else:
        features.append(False)
        features.append(False)
    features.append(link_no != query_no and common == 0)
    features.append(link_no != query_no and common == 1)
    features.append(link_no != query_no and common > 1)
    features.append(link_no != query_no and common > 5)
    
    # Convert to 0/1
    final_features = []
    for feature in features:
        if feature == True:
            final_features.append(1.0)
        elif feature == False:
            final_features.append(0.0)
        else:
            final_features.append(feature)

    if do_cache:
        cache[name, query_no, link_no] = final_features
    return final_features

def read_data(filenames, is_test=False):
    # Initialize a list to store instances and a set to track processed file names.
    instances = []
    done = set()

    # Loop over each file name to process the data.
    for filename in filenames:
        # Extract the base name of the file by removing known file type endings.
        name = filename
        for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
            if filename.endswith(ending):
                name = filename[:-len(ending)]
        # Skip processing if this file has already been processed.
        if name in done:
            continue
        # Mark this file as processed.
        done.add(name)
        # Read and split the ASCII text data into a list of words.
        text_ascii = [l.strip().split() for l in open(name + ".ascii.txt")]
        # Initialize an empty list to store tokenized text.
        text_tok = []
        # Read, clean, and tokenize the text data.
        for l in open(name + ".tok.txt"):
            l = l.strip().split()
            # Remove sentence end token if present.
            if len(l) > 0 and l[-1] == "</s>":
                l = l[:-1]
            # Ensure that each line starts with a sentence start token.
            if len(l) == 0 or l[0] != '<s>':
                l.insert(0, "<s>")
            text_tok.append(l)
        # Convert lines of text to structured info format.
        info, target_info = lines_to_info(text_ascii)

        # Initialize an empty dictionary to store link annotations.
        links = {}
        # Populate links with empty lists for test data or read from annotation file for training data.
        if is_test:
            for i in range(args.test_start, min(args.test_end, len(text_ascii))):
                links[i] = []
        else:
            # Read annotations and store them in the links dictionary.
            for line in open(name + ".annotation.txt"):
                nums = [int(v) for v in line.strip().split() if v != '-']
                links.setdefault(max(nums), []).append(min(nums))
        # Create instances by combining file name, link indices, and the processed text and info.
        for link, nums in links.items():
            instances.append((name + ".annotation.txt", link, nums, text_ascii, text_tok, info, target_info))
    # Return the list of compiled instances.
    return instances

def simplify_token(token):
    chars = []
    for char in token:
        #### Reduce sparsity by replacing all digits with 0.
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

class PyTorchModel(nn.Module):
    ''' a simplle feed forward neural network model 
    with hidden layers and a nonlinearity function, 
    a dropout layer for regularization, 
    and a final output layer to produce predictions'''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate, nonlin):
        super(PyTorchModel, self).__init__()
        # Create word embeddings if provided
        self.embeddings = None
        if args.word_vectors:
            # Initialize token_to_id dictionary and load pretrained word vectors
            self.token_to_id = {}
            pretrained = []
            with open(args.word_vectors, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    word = parts[0].lower()
                    vector = [float(v) for v in parts[1:]]
                    # Map word to its ID and add vector to pretrained list
                    self.token_to_id[word] = len(self.token_to_id)
                    pretrained.append(vector)
            num_embeddings = len(pretrained)
            embedding_dim = len(pretrained[0])
            # Create embeddings layer and copy pretrained vectors
            self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
            self.embeddings.weight.data.copy_(torch.tensor(pretrained))
            input_size += 4 * embedding_dim

        # Define the hidden layers and biases
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        # Create a list of hidden layers
        self.hidden_layers = nn.ModuleList(layers)
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Nonlinearity function
        self.nonlin = nonlin

    def forward(self, features, word_ids=None):
        if self.embeddings and word_ids is not None:
            # Convert list of word_ids to a tensor
            word_ids_tensor = torch.tensor(word_ids, dtype=torch.long).to(features.device)
            
            # Get word embeddings
            word_embeddings = self.embeddings(word_ids_tensor)
            
            # Compute maximum and mean of word embeddings across the sequence dimension
            qvec_max, _ = torch.max(word_embeddings, dim=1)
            qvec_mean = torch.mean(word_embeddings, dim=1)
            
            # Concatenate them with features along the second dimension (features dimension)
            features = torch.cat([features, qvec_max, qvec_mean], dim=1)
        
        # Apply dropout to the input features
        h = self.dropout(features)
        
        # Pass through each hidden layer and apply nonlinearity
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.apply_nonlinearity(h)
        
        # Output layer
        h = self.output_layer(h)
        return h

    def tokens_to_ids(self, tokens):
        # Convert tokens to lowercase and map them to corresponding IDs
        return [self.token_to_id.get(token.lower(), -1) for token in tokens]  # -1 for unknown tokens

    def apply_nonlinearity(self, x):
        # Apply specified nonlinearity function
        if self.nonlin == 'linear':
            return x  # No non-linearity
        elif self.nonlin == 'tanh':
            return torch.tanh(x)
        elif self.nonlin == 'cube':
            return torch.pow(x, 3)
        elif self.nonlin == 'logistic':
            return torch.sigmoid(x)
        elif self.nonlin == 'relu':
            return F.relu(x)
        elif self.nonlin == 'elu':
            return F.elu(x)
        elif self.nonlin == 'selu':
            return F.selu(x)
        elif self.nonlin == 'softsign':
            return F.softsign(x)
        elif self.nonlin == 'swish':
            return x * torch.sigmoid(x)
        else:
            raise ValueError('Unsupported nonlinearity')

def do_instance(instance, train, model, optimizer, do_cache=True):
    name, query, gold, text_ascii, text_tok, info, target_info = instance

    # Skip cases if we can't represent them
    gold = [v for v in gold if v > query - MAX_DIST]
    if len(gold) == 0 and train:
        return 0, False, query

    # Get features
    options = []
    query_ascii = text_ascii[query]
    query_tok = model.tokens_to_ids(text_tok[query])
    for i in range(query, max(-1, query - MAX_DIST), -1):
        option_ascii = text_ascii[i]
        option_tok = model.tokens_to_ids(text_tok[i])
        features = get_features(name, query, i, text_ascii, text_tok, info, target_info, do_cache)
        options.append((option_tok, features))
    gold = [query - v for v in gold]
    lengths = [len(sent) for sent in options]

    # Run computation
    example_loss, output = model(query_tok, options, gold, lengths, query)
    loss = 0.0
    if train and example_loss is not None:
        example_loss.backward()
        optimizer.update()
        loss = example_loss.scalar_value()
    predicted = output
    matched = (predicted in gold)

    return loss, matched, predicted

if __name__ == "__main__":
    # initializing the parser
    parser = argparse.ArgumentParser(description='IRC Conversation Disentangler.')
    initializing()
    FEATURES = 77
    args = parser.parse_args()
    WEIGHT_DECAY = args.weight_decay
    HIDDEN = args.hidden
    LEARNING_RATE = args.learning_rate
    LEARNING_DECAY_RATE = args.learning_decay_rate
    MOMENTUM = args.momentum
    EPOCHS = args.epochs
    DROP = args.drop
    MAX_DIST = args.max_dist
    OUTPUT = MAX_DIST + 1
    log_file = open(args.prefix +".log", 'w')
    header(sys.argv, [log_file, sys.stdout])
    
    autobatch = not args.no_autobatch
    batch_size = 1 if autobatch else 64
    
    train = []
    if args.train:
        train = read_data(args.train)
        
    dev = []
    if args.dev:
        dev = read_data(args.dev)
    if args.train:
        step = 0
        for epoch in range(EPOCHS):
            random.shuffle(train)
    test = dev
    if args.test:
        test = read_data(args.test, True)
    if args.random_sample and args.train:
        random.seed(args.seed)
        random.shuffle(train)
        train = train[:int(args.random_sample)]   

    model = None
    optimizer = None
    scheduler = None
    
    model = PyTorchModel(input_size=FEATURES, hidden_size=HIDDEN, output_size=OUTPUT, num_layers=args.layers, dropout_rate=args.drop, nonlin=args.nonlin)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LEARNING_DECAY_RATE)
    
    prev_best = None
    
    # training loop i guess
    if args.train:
        step = 0
        for epoch in range(EPOCHS):
            random.shuffle(train)
            # update the learning rate
            optimizer.learning_rate = LEARNING_RATE / (1+ LEARNING_DECAY_RATE * epoch)

             # Loop over batches
            loss = 0
            match = 0
            total = 0
            loss_steps = 0
            for instance in train:
                step += 1
            
            # torch.cuda.empty_cache()  # If you're using GPU to free up memory
            torch.autograd.set_grad_enabled(False)  # Disable gradient computation
            torch.autograd.set_grad_enabled(True)  # Enable gradient computation
            
            ex_loss, matched, _ = do_instance(instance, True, model, optimizer)
            loss += ex_loss
            loss_steps += 1
            
            if matched:
                match += 1
            total += len(instance[2])
           
            if step % args.report_freq == 0:
                # Dev pass
                dev_match = 0
                dev_total = 0
                
                for dinstance in dev: 
                    ex_loss, matched, _ = do_instance(dinstance, False, model, optimizer)
                    if matched:
                        dev_match += 1
                    dev_total += len(dinstance[2])
                
                tacc = match / total
                dacc = dev_match / dev_total
                print("{} tl {:.3f} ta {:.3f} da {:.3f} from {} {}".format(epoch, loss / loss_steps, tacc, dacc, dev_match, dev_total), file=log_file)
                log_file.flush()
                
                if prev_best is None or prev_best[0] < dacc:
                    prev_best = (dacc, epoch)
                    model.model.save(args.prefix + ".pytorch.model")
        
            if prev_best is not None and epoch - prev_best[1] > 5:
                break
            
        # Load model
        if prev_best is not None or args.model:
            location = args.model
            if location is None:
                location = args.prefix +".pytorch.model"
            model.model.populate(location)
        for instance in test:
                 _, _, prediction = do_instance(instance, False, model, optimizer, False)
                 print("{}:{} {} -".format(instance[0], instance[1], instance[1] - prediction)) 
                 
    log_file.close()  