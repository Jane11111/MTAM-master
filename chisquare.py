import scipy.stats as stats
with open('best_result_hr_movielen_MTAM', 'r') as f:
    obs =f.readline()
    obs = eval(obs)
with open('best_result_hr_movielen_T_SeqRec', 'r') as f:
    exp =f.readline()
    exp = eval(exp)
print(stats.ttest_rel(obs,exp))

