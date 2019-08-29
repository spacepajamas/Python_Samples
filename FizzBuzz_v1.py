## Goal easy to modify
param_dict = {3:'Fizz', 5:'Buzz'} ## try:  {3:'Fizz', 5:'Buzz', 7:'Bing'}, code will still work without changes
r = 200 # Optional, can be hardcoded into the range function 
for i in range(1,r):
    out = ''
    for k in param_dict.keys():
        if i%k == 0:
            out = out+param_dict[k]
    if out:
        print(out)
    else:
        print(i)
