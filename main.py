# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:41:50 2019

@author: Detlef Schmicker

licence: gplv3, see licence.txt file

"""
# pylint: disable=C0103, C0301, C0111, R0903, E0012, C1801, R0205, R1705

# pylint: disable=W0622
# for usage with python2 and python3
from __future__ import print_function    # (at top of module)
from random import random
from random import randint
from builtins import input
from past.builtins import basestring    # pip install future
import numpy as np

# pylint: enable=W0622

# uncomment to add some editing features to the input command
# import readline #@UnusedVariable


only_one_answer = True

trace_on = False

#creats vars
class var(object): # prolog variable
    pass
#    def __init__(self, name=None):
#        self.name = name

#marks lists
class l(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B

#marks rules
class rule(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B

#marks empty list
class empty(object):
    pass

#import copy
empty_list = empty #this is a reference to the class, this way deepcopy would work too

#marks cut predicate
class cut(object):
    pass

class repeat(object):
    pass

class rnn(object):
    def call_rnn(self, term, mode, bounds):
        term_str = formatl(term, bounds, {})
        #print(term_str)
        model_input = str_to_int_list(term_str)
        #print(model_input)
        tmp_x = np.array([model_input], dtype=int).reshape((1, -1))
        prediction = model.predict(tmp_x)[0]
        #print(prediction)
        if mode == 'best':
            predict_pos = np.argmax(prediction)
            yield predict_pos
        elif mode == 'order':
            predict_sort = np.argsort(-prediction)
            for xx in predict_sort:
                yield xx
        else:
            sum_prediction = prediction.sum()
            num_prediction = prediction.size
            factor = float(mode) / 100 # in percent
            sum_prediction += factor
            factor /= num_prediction
            rand = sum_prediction * random()
            sss = 0
            #print(rand, sum_prediction, factor)
            #for i in range(num_prediction): print('{:6.3f}'.format(prediction[i]), end='')
            #print()
            ret = num_prediction-1
            for i in range(num_prediction):
                sss += prediction[i] + factor
                #print(sss,i)
                if rand < sss:
                    ret =  i
            yield ret

    def calculate(self, calc_object, bounds):
        # This calculates recursively the int result of the list object
        calc_object = final_bound(calc_object, bounds)
        op1 = calc_object.A
        op2 = calc_object.B.A
        for xx in self.call_rnn(op1, op2, bounds):
            yield True, xx
        yield False, '0'

    def do_calc(self, term, bounds):
        calc_object = term.B.A
        for (t, result) in self.calculate(calc_object, bounds):
            if t:
                t1, new_bounds = match(term.A, str(result), bounds)
                if t1:
                    #new_bounds.update(bounds)
                    yield True, new_bounds
            yield False, bounds


#marks calc for is
class calc(object):
    # pylint: disable=R0911, R0912
    def calculate(self, calc_object, bounds):
        # This calculates recursively the int result of the list object
        calc_object = final_bound(calc_object, bounds)
        # it is an object
        if isinstance(calc_object, basestring):
            return True, calc_object
        else:
            op = calc_object.A
            op1 = calc_object.B.A
            op2 = calc_object.B.B.A
            if op == 'write': #used for changing to much and allow writing to files add .tmp for security reasons
                # op1 is filename, op2 is object to print
                F = open(final_bound(op1, bounds)+".tmp", 'a')
                if op2 == 'track': # if the second object is track, the track_for_ai list is written to the file
                    F.write(str(track_for_ai[0]) + "\n")
                else:
                    F.write(formatl(op2, bounds, {})+"\n")
                F.close()
                return True, str(1) #return 1, no sense in real calculations !!
            elif op == 'rnn':
                # op1 is the structure,to check
                # op2 allows to manipulate the mode
                #       eg. random with probability     ['rnn',X,'random']
                #           best (no probability)       ['rnn',X,'best']
                #
                # returns 0 - max_output
                return True, call_rnn(op1, op2, bounds)

            t, op1 = self.calculate(calc_object.B.A, bounds)
            if t:
                t2, op2 = self.calculate(calc_object.B.B.A, bounds)
                if t2:
                    if op == 'add':
                        return True, str(int(op1)+int(op2))
                    elif op == 'mult':
                        return True, str(int(op1)*int(op2))
                    elif op == 'sub':
                        return True, str(int(op1)-int(op2))
                    elif op == 'div':
                        return True, str(int(op1)/int(op2))
                    elif op == 'mod':
                        return True, str(int(op1) % int(op2))
                    elif op == 'lower':
                        if int(op1) < int(op2):
                            return True, str(1)
                        return False, str(0)
                    elif op == 'lowereq':
                        if int(op1) <= int(op2):
                            return True, str(1)
                        return False, str(0)
                    elif op == 'neq':
                        if int(op1) != int(op2):
                            return True, str(1)
                        return False, str(0)
                    elif op == 'eq':
                        if int(op1) == int(op2):
                            return True, str(1)
                        return False, str(0)
                    elif op == 'rand':
                        return True, randint(int(op1), int(op2))

            return False, calc_object

    def do_calc(self, term, bounds):
        calc_object = term.B.A
        t, result = self.calculate(calc_object, bounds)
        if t:
            t1, new_bounds = match(term.A, str(result), bounds)
            if t1:
                #new_bounds.update(bounds)
                return True, new_bounds
        return False, bounds

def final_bound(A, bounds):
    if not isinstance(A, var):
        return A # None is not allowed as prolog object !!!
    if A in bounds:
        return final_bound(bounds[A], bounds)
    return A

def get_new_var(name, local_vars):
    if name not in local_vars:
        local_vars[name] = var()
    return local_vars[name]

# all vars get new instances
def renew_vars(line, local_vars):
    #return copy.deepcopy(line) # the original impementaion was faster, this is shorter :)
    # pylint: disable=R0911
    if isinstance(line, l):
        return l(renew_vars(line.A, local_vars), renew_vars(line.B, local_vars))
    elif isinstance(line, var):
        return get_new_var(line, local_vars)
    elif isinstance(line, rule):
        return rule(renew_vars(line.A, local_vars), renew_vars(line.B, local_vars))
    elif isinstance(line, list):
        if len(line) == 0:
            return []
        return [renew_vars(line[0], local_vars)] + renew_vars(line[1:], local_vars)
    return line
#    elif isinstance(line, basestring):
#        return line
#    elif line == empty_list: #isinstance(line, empty):
#        return empty_list
#    raise Exception("clause with illegal structure " + str(line))

def check_if_var_in_object(final_var, final_other_in, bounds):
    final_other = final_bound(final_other_in, bounds)
    if final_var == final_other:
        return True
    if isinstance(final_other, l):
        if check_if_var_in_object(final_var, final_other.A, bounds):
            return True
        if check_if_var_in_object(final_var, final_other.B, bounds):
            return True

    return False

#vvv={}
# returns True or False, and the new bounds in case of True, otherwize the old ones
def match(A, B, bounds):
    # pylint: disable=R0911
    final_A = final_bound(A, bounds)
    final_B = final_bound(B, bounds)
    if final_A == final_B:
        return True, bounds
    if isinstance(final_B, var): # not bound
        if isinstance(final_A, var) or not check_if_var_in_object(final_B, final_A, bounds):
            new_bounds = bounds.copy()
            new_bounds[final_B] = final_A
            return True, new_bounds
        return False, bounds
    else:
        if isinstance(final_A, var):
            if not check_if_var_in_object(final_A, final_B, bounds): # isinstance(final_B, var) not possible is gurantied
                new_bounds = bounds.copy()
                new_bounds[final_A] = final_B
                return True, new_bounds
            return False, bounds
        else:
            if isinstance(final_A, l) and isinstance(final_B, l):
                t1, b2 = match(final_A.A, final_B.A, bounds)
                if t1:
                    t2, b3 = match(final_A.B, final_B.B, b2)
                    if t2:
                        return True, b3
            return False, bounds


assertz_data = {}

def assertz(predicate, infolist):
    if predicate in assertz_data:
        assertz_data[predicate].append(infolist)
    else:
        assertz_data[predicate] = [infolist]

def add_rule(predicate, infolist, tail): # tail is a list with c(predicate, infolist)
    if predicate in assertz_data:
        assertz_data[predicate].append(rule(infolist, tail))
    else:
        assertz_data[predicate] = [rule(infolist, tail)]

#make a string from a prolog l object (list)print "calc",contains.A, final_bound(contains.A, new_bounds)
def formatl(X_orig, bounds, var_nums):
    X = final_bound(X_orig, bounds)
    ret = ""
    komma = ""
    closeb = ""
    if isinstance(X, empty):
        return "[]"
    if isinstance(X, l):
        ret = "["
        closeb = "]"
    while isinstance(X, l):
        ret += komma + formatl(X.A, bounds, var_nums)
        X = final_bound(X.B, bounds)
        komma = ","
    if isinstance(X, list):
        ret += "["
        koma2 = ""
        for e in X:
            ret += koma2 + formatl(e, bounds, var_nums)
            koma2 = ","
        return ret + "]"
    if isinstance(X, var):
        v_num = 0
        if X in var_nums:
            v_num = var_nums[X]
        else:
            v_num = len(var_nums)
            var_nums[X] = v_num
        ret += komma+"_"+str(v_num)
    elif X is not None and X != empty_list:
        ret += komma+str(X)
    return ret+closeb

def ask_list(list_of_calls, bounds, cut_count):
    first = list_of_calls[0]
    rest = list_of_calls[1:]
    xx = ask(first.A, first.B, bounds, cut_count)
    for x0 in xx:
        t, new_bounds = x0
        if trace_on:
            mark = "#f#"
            if t:
                mark = "#t#"
            print(mark,first.A,formatl(first.B, new_bounds, {}))
        if t:
            if len(rest) > 0:
                xxx = ask_list(rest, new_bounds, cut_count)
                for x1 in xxx:
                    yield x1
            else:
                yield t, new_bounds
    yield False, bounds


#it is orderd by predicate answerd. The first is the predicate wich is answerd first.
track_for_ai = [[]]
limit_recursion_with_track_for_ai_length = [9999999]

#cut_count variable, and all fail, if >1 ?
#generates the solutions
def ask(predicate, infolist, bounds, cut_count):
    # pylint: disable=R0101, R0912, R0915, R0914
    #print(predicate, formatl(infolist,bounds,{}))
    if track_for_ai and limit_recursion_with_track_for_ai_length and len(track_for_ai[0]) > limit_recursion_with_track_for_ai_length[0]:
        yield False, bounds

    elif cut_count[0] > 1:
        yield False, bounds #cut accured
    elif predicate not in assertz_data:
        print("Warning -- no clause for ", predicate)
        yield False, bounds
    else:
        if track_for_ai:
            local_track = track_for_ai[0][:] # copy
            local_count = 0
        contains = assertz_data[predicate]
        if isinstance(contains, calc): #is predicate
            t, new_bounds = contains.do_calc(infolist, bounds)
            if t:
                if track_for_ai:
                    if local_count > 0:
                        track_for_ai[0] = track_for_ai[0][:-1]
                    local_count += 1
                    track_for_ai[0] += [(predicate, formatl(infolist, bounds, {}))]
                yield t, new_bounds
            else:
                if track_for_ai:
                    track_for_ai[0] = local_track
                yield False, bounds
        elif isinstance(contains, rnn): #is predicate
            for (t, new_bounds) in contains.do_calc(infolist, bounds):
                if t:
                    if track_for_ai:
                        if local_count > 0:
                            track_for_ai[0] = track_for_ai[0][:-1]
                        local_count += 1
                        track_for_ai[0] += [(predicate, formatl(infolist, bounds, {}))]
                    yield t, new_bounds
                else:
                    if track_for_ai:
                        track_for_ai[0] = local_track
                    yield False, bounds
        elif isinstance(contains, cut): #cut predicate
            cut_count[0] = 0 # the last cut overwrites all other cuts in the rule
            for xx in iter([(True, bounds), (False, bounds)]):
                cut_count[0] += 1 #returning True and False and increasing count every time
                if track_for_ai:
                    t_ai, _ = xx
                    if t_ai:
                        if local_count > 0:
                            track_for_ai[0] = track_for_ai[0][:-1]
                        local_count += 1
                        track_for_ai[0] += [(predicate, 0)]
                    else:
                        track_for_ai[0] = local_track
                yield xx
        elif isinstance(contains, repeat):
            how_often = int(final_bound(infolist.A, bounds))
            if how_often == 0:
                while 1:
                    yield True, bounds
            else:
                for _ in range(how_often):
                    yield True, bounds
                yield False, bounds
        else:
            for lll in contains:
                # check if correct to yield every output ??? only the last??
                cut_count_local = [0]
                if track_for_ai:
                    local_count += 1
                line = renew_vars(lll, {})
                if isinstance(line, rule):
                    t, new_bounds = match(infolist, line.A, bounds)
                    if t:
                        xx = ask_list(line.B, new_bounds, cut_count_local)
                        if track_for_ai:
                            last_bounds = new_bounds
                        for x0 in xx:
                            if cut_count_local[0] > 1:
                                break
                            if track_for_ai:
                                t_ai, b_ai = x0
                                if t_ai:
                                    if local_count > 1:
                                        track_for_ai[0] = track_for_ai[0][:-1]
                                    track_for_ai[0] += [(predicate, local_count, formatl(line.A, last_bounds, {}))]
                                    last_bounds = b_ai
                                else:
                                    track_for_ai[0] = local_track
                            yield x0
                    if track_for_ai:
                        track_for_ai[0] = local_track
                    if cut_count_local[0] > 1:
                        break
                    yield False, bounds
                else: #fact
                    t, new_bounds = match(infolist, line, bounds)
                    if t:
                        if track_for_ai:
                            if local_count > 1:
                                track_for_ai[0] = track_for_ai[0][:-1]
                            track_for_ai[0] += [(predicate, local_count, formatl(infolist, bounds, {}))]
                        yield True, new_bounds
                    else:
                        if track_for_ai:
                            track_for_ai[0] = local_track
                        yield False, bounds

def ask_print(predicate, infolist, bounds, wait_for_enter):
    if track_for_ai:
        track_for_ai[0] = []
    xx = ask(predicate, infolist, bounds, [0])
    for t, new_bounds in xx:
        if t:
            if track_for_ai:
                print("track_for_ai", track_for_ai[0])
            #print(formatl(infolist, new_bounds, vvv))
            print(formatl(infolist, new_bounds, {}))
            if wait_for_enter:
                cc = input('. (stop) a (all):')
                if cc == 'a':
                    wait_for_enter = False
                if cc == '.':
                    break
            elif only_one_answer:
                break # only one result

# pylint: disable=C0413, C0411
# modified from PyLog
from pyparsing import (Group, Keyword, NoMatch, Suppress, Word, ZeroOrMore, Forward, nestedExpr,
                       ParseException, alphas)
# pylint: enable=C0413
def parse_imp(iii):
    #pylint: disable= R0914
    # Grammar:
    #
    # <expr> ::= <integer>
    #            true
    #            false
    #            <identifier>
    #
    # <fact> ::= relation(constant, ...).
    #
    # <rule> ::= name(constant, ... ) :- name(constant, ...), ... .
    #

    idChars = alphas + "_+*!=<>"

    pNAME = Word(idChars + "0123456789")


    pOBJECT = Forward()
    nestedBrackets = nestedExpr('[', ']', content=pOBJECT)

# pylint: disable=W0106
# dont know why pylint warning, line is from example code
    pOBJECT << (pNAME | (Suppress(',') | '|') | nestedBrackets)
    pOBJECT.setParseAction(lambda result: result)
# pylint: enable=W0106

    pRELATION = pNAME + "(" + Group(pOBJECT + ZeroOrMore(Suppress(",") + pOBJECT)) + ")"
    pRELATION.setParseAction(lambda result: (result[0], result[2]))

    pDECL_RELATION = pRELATION + "."
    pDECL_RELATION.setParseAction(lambda result: result[0])

    pTOP_QUERY = pRELATION + "?"
    pTOP_QUERY.setParseAction(lambda result: {"result": "query", "stmt": result[0]})

    pTOP_RULE = pRELATION + ":-" + Group(pRELATION + ZeroOrMore(Suppress(",") + pRELATION)) + "."
    pTOP_RULE.setParseAction(lambda result: {"result": "rule", "rule": result})

    pTOP_FACT = (pDECL_RELATION ^ NoMatch())
    pTOP_FACT.setParseAction(lambda result: {"result": "fact", "decl": result[0]})

    pQUIT = Keyword("#quit")
    pQUIT.setParseAction(lambda result: {"result": "quit"})

    pCLEAR = Keyword("#clear")
    pCLEAR.setParseAction(lambda result: {"result": "clear"})

    pLOAD = Keyword("#load") + pNAME
    pLOAD.setParseAction(lambda result: {"result": "load", "file": result[1]})

    pLOADRNN = Keyword("#loadrnn") + pNAME
    pLOADRNN.setParseAction(lambda result: {"result": "loadrnn", "file": result[1]})

    pTRACE = Keyword("#trace")
    pTRACE.setParseAction(lambda result: {"result": "trace"})

    pTOP = (pQUIT ^ pCLEAR ^ pTRACE ^ pLOAD ^ pLOADRNN  ^ pTOP_RULE ^ pTOP_FACT ^ pTOP_QUERY)

    result = pTOP.parseString(iii)[0]
    return result

def create_list(inlist, local_vars):
    if len(inlist) == 0:
        return empty_list
    o = inlist[0]
    if o == '|':
        #restlist
        rest_variable = inlist[1]
        if rest_variable[0].isupper():
            return get_new_var(rest_variable, local_vars)
        print("list rest must be a variable:", rest_variable)
        return None
    if isinstance(o, list):
        return l(create_list(o, local_vars), create_list(inlist[1:], local_vars))
    if isinstance(o, basestring) and o[0].isupper():
        o = get_new_var(o, local_vars)
    return l(o, create_list(inlist[1:], local_vars))

def create_l(inlist, local_vars):
    if inlist == []:
        return empty_list
    o = inlist[0]
    if isinstance(o, list):
        o = create_list(o, local_vars)
    if isinstance(o, basestring) and o[0].isupper():
        o = get_new_var(o, local_vars)
    return l(o, create_l(inlist[1:], local_vars))

def imp(iii, wait_for_enter=False):
    global trace_on
    if iii.strip() == '':
        return True
    local_vars = {}
    ii = parse_imp(iii)
    if ii['result'] == 'fact':
        decl = ii['decl']
        predicate = decl[0]
        assertz(predicate, create_l(decl[1].asList(), local_vars))
    elif ii['result'] == 'query':
        query = ii['stmt']
        predicate = query[0]
        ask_print(predicate, create_l(query[1].asList(), local_vars), {}, wait_for_enter)
    elif ii['result'] == 'rule':
        rrr = ii['rule']
        predicate = rrr[0][0]
        left_side = rrr[0][1].asList()
        right_side = []
        for rr in rrr[2]:
            onecall = [rr[0]] + rr[1].asList()
            right_side.append(create_l(onecall, local_vars))
        assertz(predicate, rule(create_l(left_side, local_vars), right_side))
    elif ii['result'] == 'quit':
        return False
    elif ii['result'] == 'load':
        load_file(ii['file']+'.pl')
    elif ii['result'] == 'loadrnn':
        setup_rnn(ii['file']+'.hdf5')
    elif ii['result'] == 'clear':
        init_data()
    elif ii['result'] == 'trace':
        trace_on = not trace_on
        print("Trace now", trace_on)
    return True

def load_file(f):
    ff = open(f)
    for line in ff:
        print("#loaded: ", line.strip())
        imp(line)

def prolog():
    print("\n#quit to leave prolog, #clear to clean database #load to load a file (adding ending .pl)")
    print("atom( .... ) format is only allowed for predicates, if you want to have add mult use lists with [add,X,Y]")
    print("also for structures use [ .....  ], internally it is transfered to [x0,x1] -> l(x0,l(x1,None))")
    while 1:
        command = input("PyProlog==> ")
        try:
            t = imp(command, True)
            if not t:
                break

        except RuntimeError as re:
            print("Runtime Error: ", re)
        except ParseException as re:
            print("ParseExeption: ", re)

#fresh database, adding build in predicates cut and is
def init_data():
    assertz_data.clear()
    assertz_data['is'] = calc()
    assertz_data['rnn'] = rnn()
    assertz_data['cut'] = cut()
    assertz_data['repeat'] = repeat()


# helper functions for AI
def print_assertz_data():
    for predicate in assertz_data:
        if predicate not in ('is', 'cut'):
            for fact_or_rule in assertz_data[predicate]:
                if isinstance(fact_or_rule, rule):
                    print(predicate, formatl(fact_or_rule.A, {}, {}), formatl(fact_or_rule.B, {}, {}))
                else:
                    print(predicate, formatl(fact_or_rule, {}, {}))


###############################################################################
# Neural Network
#
###############################################################################
model = None    # pretrained keras model
ml = 0          # length of onput string

#setup vocabulary
vocab = {}
count_chars = 0
def add_translate(cc):
    #pylint: disable=W0603
    global count_chars
    vocab[cc] = count_chars
    count_chars += 1

for c in range(ord('a'), ord('z')+1):
    add_translate(chr(c))
for c in range(ord('0'), ord('9')+1):
    add_translate(chr(c))

add_translate(',')
add_translate('[')
add_translate(']')
add_translate('_')
add_translate(' ')

def str_to_int_list(x):
    # uncomment for all the same length
    # x = x[::-1]
    x = x[-ml:] #if to long only take the last ml characters
    x = ('{:>'+str(ml)+'}').format(x)
    ret = []
    for cc in x:
        ret.append(vocab[cc])
    return ret

def call_rnn(term, mode, bounds):
    term_str = formatl(term, bounds, {})
    #print(term_str)
    model_input = str_to_int_list(term_str)
    #print(model_input)
    tmp_x = np.array([model_input], dtype=int).reshape((1, -1))
    prediction = model.predict(tmp_x)[0]
    #print(prediction)
    if mode == 'best':
        predict_pos = np.argmax(prediction)
        return predict_pos
    else:
        sum_prediction = prediction.sum()
        num_prediction = prediction.size
        factor = float(mode) / 100 # in percent
        sum_prediction += factor
        factor /= num_prediction
        rand = sum_prediction * random()
        sss = 0
        #print(rand, sum_prediction, factor)
        #for i in range(num_prediction): print('{:6.3f}'.format(prediction[i]), end='')
        #print()
        for i in range(num_prediction):
            sss += prediction[i] + factor
            #print(sss,i)
            if rand < sss:
                return i
        return num_prediction-1

def setup_rnn(model_name):
    #pylint: disable=W0603
    global model, ml
    from keras.models import load_model
    model = load_model(model_name)
    ml = model.layers[0].input_shape[1]


###############################################################################



try:
# pylint: disable=W0122, W0703
    exec(open('config.py').read())
except Exception as ee:
    print("no config.py file loaded", ee)

# execute a test before
init_data()
load_file('test.pl')

setup_rnn('final_model.hdf5')

#print_assertz_data()

# start prolog promt
init_data()
prolog()
#load_file('deb.pl')


# Examples which should work
#
# is(A,[mult,3,3])?
#
# #load test
#
# cut(1) because of parsing cut must contain anything in the brackets !!
