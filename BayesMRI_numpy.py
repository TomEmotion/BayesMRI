#!/usr/bin/python

# BayesMRInumpy.py: Calculate the Bayes Factor associated with a given group-level fMRI t-stat map
# This version makes use of the NumPy and NiBabel packages to perform image calculations without FSL
# Written by Tom Johnstone (2016) itjohnstone@gmail.com
# Based on code provided by Zoltan_Dienes (http://www.lifesci.sussex.ac.uk/home/Zoltan_Dienes/inference/Bayes.htm)


import os,re,getopt,sys,math,subprocess,numpy,time
import nibabel as nib
from scipy.stats import norm

def nifti_check_file(filename):
    filename,theext = os.path.splitext(filename)
    if theext == ".gz":
        filename,theext = os.path.splitext(filename)
        theext = theext + ".gz"
    if theext == ".nii":
        filename = filename+".nii"
        return filename,os.access(filename,os.R_OK)
    elif theext == ".nii.gz":
        filename = filename+".nii.gz"
        return filename,os.access(filename,os.R_OK)
    elif theext == "":
        filename = filename+".nii"
        if os.access(filename,os.R_OK):
            return filename,os.access(filename,os.R_OK)
        else:
            filename = filename+".nii.gz"
            return filename,os.access(filename,os.R_OK)
    else:
         return filename,False       
 
def main():
    
    start_time = ticks = time.clock()
    
    out_file = ""
    iterations = 1000
    opts,args = getopt.getopt(sys.argv[1:],')ht:c:d:o:i:u:p:n:')

    for opt,param in opts:
     if opt == '-h':
         print ('\nbayesMRI_numpy.py\n')
         print ('This program calculates the Bayes Factor associated with a given group level fMRI t-test. You can specify ')
         print ('a global uniform prior, a global normal prior, or a spatially varying normal prior based on a previous analysis. ')
         print ('For the normal priors, probabilities will be integrated between +/- 5 standard errors of the mean\n')
         print ('The program outputs the Bayes Factor and log Bayes Factor for the model compared to the null (as well as the inverse).')
         print ('Likelihoods for the model and null are also output.\n')
         print ('This program requires Python with NumPy and NiBabel packages installed\n')
         print ('Usage:\n\t')
         print ('BayesMRI_numpy.py -t <tstat_file> -c <contrast_file> -d <dof_file> [options]\n')
         print ('\ttstat_file: tstat file from group analysis')
         print ('\tcontrast_file: contrast (cope) file from group analysis')
         print ('\ttdof_file: file containing degrees of freedom for the group analysis\n')
         print ('Options:')
         print ('\t-o <ouput_file_stem> prefix for output file names (default = "")\n')
         print ('\t-i <iterations> the number of bins used to integrate probabilities (larger = more precise; default = 1000)\n')
         print ('\t-u <lower_limit, upper_limit> Uniform prior: a uniform distribution between lower_limit and upper_limit\n')
         print ('\t-n <mean,stderr> Normal (Gaussian) prior: a normal distribution with given mean and standard error\n')
         print ('\t-p <contrast_file,tstat_file> Data-dependent prior: a spatially varying Gaussian prior with voxelwise mean')
         print ('\t   and standard error based on a previous analysis\n')
         print ('\t-h this information\n\n')
         sys.exit()
     elif param != '':
         if opt == '-t':
             tstat_file, file_ok = nifti_check_file(param)
             if not file_ok:
                 print ("Error: Couldn't read t-stat file: "+tstat_file)
                 sys.exit(1)
             tstatfile = nib.load(tstat_file)
         if opt == '-c':
              cope_file, file_ok = nifti_check_file(param)
              if not file_ok:
                  print ("Error: Couldn't read contrast file: "+cope_file)
                  sys.exit(1)
              copefile = nib.load(cope_file)
         if opt == '-d':
             dof_file, file_ok = nifti_check_file(param)
             if not file_ok:
                 print ("Error: Couldn't read DOF file: "+dof_file)
                 sys.exit(1)
             doffile = nib.load(dof_file)
         elif opt == '-o':
             out_file = param+"_"
         elif opt == '-i':
             iterations = int(param)
         elif opt == '-u':
             prior = "uniform"
             [lowerstr,upperstr] = param.split(',')
             priorlower = float(lowerstr)
             priorupper = float(upperstr)
         elif opt == '-n':
             prior = "normal"
             [meanstr,sdstr] = param.split(',')
             priormean = float(meanstr)
             priorsd = float(sdstr)
             priorvar = numpy.square(priorsd)
         elif opt == '-p':
             prior = "dataprior"
             print (prior)
             [priorcopefile,priortstatfile] = param.split(',')
             if not (os.access(priorcopefile+".nii.gz",os.R_OK) and os.access(priortstatfile+".nii.gz",os.R_OK)):
                 print ("Error: Couldn't read prior cope or tstat files: "+priorcopefile,priortstatfile)
                 sys.exit(1)
             prior_copefile = nib.load(priorcopefile+".nii.gz")
             prior_tstatfile = nib.load(priortstatfile+".nii.gz")
             priormean = prior_copefile.get_data()
             priortstat = prior_tstatfile.get_data()
             priorvar = numpy.square(priormean / priortstat)
     else:
         print ('Error: Option '+opt+' not recognised. Use -h option for usage')
         sys.exit()

    cope = copefile.get_data()
    tstat = tstatfile.get_data()
    stderr = cope / tstat
    dof = doffile.get_data()
    sd2 = numpy.square(stderr * (1 + 20 / numpy.square(dof)))
    likelyhoodTheory = stderr * 0
    denom = numpy.sqrt(sd2 * 2 * numpy.pi)

    if prior == "uniform":
        disttheta = 1.0/(priorupper-priorlower)
    elif prior == "normal" or prior == "dataprior":
        priorlower = priormean - numpy.sqrt(priorvar) * 5
        priorupper = priormean + numpy.sqrt(priorvar) * 5
        distdenom = numpy.sqrt(2 * numpy.pi * priorvar)
    else:
        print ('Error: You must specify a permitted prior distribution. Use -h option for usage')
        sys.exit()
    
    incr = (priorupper-priorlower)/iterations
        
    for A in range(0,iterations):
        if A/10.0 == round(A/10.0):
            print ("iteration: " + str(A))
        theta = priorlower + A*incr    
        if prior == "normal" or prior == "dataprior":
            disttheta = norm_pdf(priormean,priorvar,theta,distdenom)
            #disttheta = norm(priormean,priorvar).pdf(theta) #need to change the variance to sd if using this function
        likelyhoodTheory = likelyhoodTheory + norm_pdf(theta, sd2, cope, denom) * disttheta * incr
        #likelyhoodTheory = likelyhoodTheory + norm(theta, sd2).pdf(cope) * disttheta * incr #need to change the variance to sd if using this function

    likelyhoodNull = norm_pdf(0, sd2, cope, denom)
    #likelyhoodNull = norm(0, sd2).pdf(cope) #need to change the variance to sd if using this function
    
    bayesFactorModel = likelyhoodTheory / likelyhoodNull
    bayeslogFactorModel = numpy.log10(bayesFactorModel)
    bayesFactorNull = 1 / bayesFactorModel
    bayeslogFactorNull = numpy.log10(bayesFactorNull)
    
    imgs = nib.Nifti1Image(likelyhoodTheory, copefile.affine, copefile.header)
    nib.save(imgs, out_file+'likelyhoodTheory.nii.gz')
    imgs = nib.Nifti1Image(likelyhoodNull, copefile.affine, copefile.header)
    nib.save(imgs, out_file+'likelyhoodNull.nii.gz')
    imgs = nib.Nifti1Image(bayesFactorModel, copefile.affine, copefile.header)
    nib.save(imgs, out_file+'bayesFactorModel.nii.gz')
    imgs = nib.Nifti1Image(bayeslogFactorModel, copefile.affine, copefile.header)
    nib.save(imgs, out_file+'bayeslogFactorModel.nii.gz')
    imgs = nib.Nifti1Image(bayesFactorNull, copefile.affine, copefile.header)
    nib.save(imgs, out_file+'bayesFactorNull.nii.gz')
    imgs = nib.Nifti1Image(bayesFactorNull, copefile.affine, copefile.header)
    nib.save(imgs, out_file+'bayeslogFactorNull.nii.gz')
    
    print ("The likelihood of your data given your theory is in the file: "+out_file+"likelihoodTheory")
    print ("The likelihood of your data given the null is in the file: "+out_file+"likelihoodNull")
    print ("The Bayes Factor for the model versus the null is in the file: "+out_file+"bayesFactorModel")
    print ("The Bayes Factor for the null versus the model is in the file: "+out_file+"bayesFactorNull")
    print ("The log (base 10) Bayes Factor for the model versus the null is in the file: "+out_file+"bayeslogFactorModel")
    print ("The log (base 10) Bayes Factor for the null versus the model is in the file: "+out_file+"bayeslogFactorNull")
    
    end_time = ticks = time.clock()
    time_taken = end_time - start_time
    print ("Total time taken: "+str(time_taken)+" seconds")


# Define own version of normal pdf so that the constant denominator doesn't need to be recalculated for each iteration (for speed)
# The script runs about 5 times faster when using this function in place of the built in function
def norm_pdf(mean, variance, x, thedenom):
    # for the normal distribution with mean mn and variance given in varianceFile, return the Y value (probability) corresponding to value X
    return numpy.exp(-numpy.square(x - mean) / (2 * variance)) / thedenom


if __name__ == "__main__":
    main()

