import numpy as np
def prepare_params(dicparams):
    params=[]
    params_dic={}
    p0=[]
    dp=[]
    bounds=[]
    n=0
    #slices={}
    Rpa=np.array([])
    Ri=np.array([])
    minpa=100
    maxpa=0
    mini=100
    maxi=0
    
    paf=np.array([],dtype=bool)
    pa0=np.array([])
    iif=np.array([],dtype=bool)
    i0=np.array([])
    parnames=[]
    for p in dicparams:
        k0=n
        for i in dicparams[p]:
            if i>0:
                params_dic.update({p+'_'+str(i):dicparams[p][i][0]})
            else:
                params_dic.update({p:dicparams[p][i][0]})
#             if p == 'pa':
#                 Rpa=np.append(Rpa,i)
            if p == 'pa':
                pa0=np.append(pa0,dicparams[p][i][0])
                paf=np.append(paf,dicparams[p][i][3])
                Rpa=np.append(Rpa,i)
            elif p == 'i':
                i0=np.append(i0,dicparams[p][i][0])
                iif=np.append(iif,dicparams[p][i][3])
                Ri=np.append(Ri,i)     
            if dicparams[p][i][3]:
                if p == 'pa':
                    if n<minpa: minpa=n
                    if n>maxpa: maxpa=n
                elif p == 'i':
                    if n<mini: mini=n
                    if n>maxi: maxi=n
                else:
                    print(f"Parameter {p} has index {n}")
                if i>0:
                    parnames.append(p+'_'+str(i))
                else:
                    parnames.append(p)
                params.append(dicparams[p][i][0])
                bounds.append(dicparams[p][i][1])
                p0.append(dicparams[p][i][2][0])
                dp.append(dicparams[p][i][2][1])
                n+=1
        #slices.update({p:slice(k0,n)})
    print(f"PA parameters have indexes [{minpa}:{maxpa+1}]")
    print(f"Inc parameters have indexes [{mini}:{maxi+1}]")
    print(f'Total Parameters we are going to fit {n}')
    return {'params0':params,'param_names':parnames,'bounds':np.array(bounds),'p0':np.array(p0),
          'dp':np.array(dp),'params_dic':params_dic,'n':n,'pa':pa0,'Rpa':Rpa,'i':i0,'Ri':Ri,
          'paf':paf,'iif':iif}