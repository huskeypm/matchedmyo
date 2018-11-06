#!/usr/bin/env python
import os
import sys
import time

import cv2
import imutils
import matplotlib.pyplot as plt
#
# ROUTINE  
#
import numpy as np
import pandas as pd

import bankDetect as bD
import detect
import display_util as du
import matchedFilter as mF
import optimizer
import painter
import preprocessing as pp
import twoDtense as tdt
import tissue
import util

##################################
#
# Revisions
#       July 24,2018 inception
#
##################################

class empty:pass

### Change default matplotlib settings to display figures
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 'large'
print "Comment out for HESSE"
#plt.rcParams['axes.labelpad'] = 12.0 
plt.rcParams['figure.autolayout'] = True

#root = "myoimages/"
root = "/net/share/dfco222/data/TT/LouchData/processedWithIntelligentThresholding/"

def WT_results(): 
  root = "/net/share/dfco222/data/TT/LouchData/processedMaskedNucleus/"
  testImage = root+"Sham_M_65_nucleus_processed.png"

  rawImg = util.ReadImg(testImage,cvtColor=False)

  iters = [-25,-20, -15, -10, -5, 0, 5, 10, 15, 20,25]
  coloredImg, coloredAngles, angleCounts = giveMarkedMyocyte(testImage=testImage,
                        returnAngles=True,
                        iters=iters,
                        tag='fig3',
                        writeImage = True
                        )
  correctColoredAngles = switchBRChannels(coloredAngles)
  correctColoredImg = switchBRChannels(coloredImg)

  ### make bar chart for content
  wtContent, ltContent, lossContent = assessContent(coloredImg,testImage)
  normedContents = [wtContent, ltContent, lossContent]

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  ### make a single bar chart
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, normedContents[0], width, color=colors[0])
  rects2 = ax.bar(indices+width, normedContents[1], width, color=colors[1])
  rects3 = ax.bar(indices+2*width, normedContents[2], width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  plt.gcf().savefig('fig3_BarChart.pdf',dpi=300)
  plt.close()

  ### save files individually and arrange using inkscape
  plt.figure()
  plt.imshow(switchBRChannels(util.markMaskOnMyocyte(rawImg,testImage)))
  plt.gcf().savefig("fig3_Raw.pdf",dpi=300)

  plt.figure()
  plt.imshow(correctColoredAngles)
  plt.gcf().savefig("fig3_ColoredAngles.pdf",dpi=300)

  ### save histogram of angles
  giveAngleHistogram(angleCounts,iters,"fig3")

def HF_results(): 
  '''
  TAB results
  '''
  ### initial arguments
  filterTwoSarcSize = 25
  imgName = root + "HF_1_processed.png"
  rawImg = util.ReadImg(imgName)
  markedImg = giveMarkedMyocyte(testImage=imgName,tag='fig4',writeImage=True)

  ### make bar chart for content
  wtContent, ltContent, lossContent = assessContent(markedImg,imgName)
  normedContents = [wtContent, ltContent, lossContent]

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  ### opting to make a single bar chart
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, normedContents[0], width, color=colors[0])
  rects2 = ax.bar(indices+width, normedContents[1], width, color=colors[1])
  rects3 = ax.bar(indices+2*width, normedContents[2], width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  plt.gcf().savefig('fig4_BarChart.pdf',dpi=300)
  plt.close()
 
  switchedImg = switchBRChannels(markedImg)

  plt.figure()
  plt.imshow(switchBRChannels(util.markMaskOnMyocyte(rawImg,imgName)))
  plt.gcf().savefig("fig4_Raw.pdf",dpi=300)

def MI_results(): 
  '''
  MI Results
  '''
  filterTwoSarcSize = 25

  ### Distal, Medial, Proximal
  DImageName = root+"MI_D_76_processed.png"
  MImageName = root+"MI_M_45_processed.png"
  PImageName = root+"MI_P_16_processed.png"

  imgNames = [DImageName, MImageName, PImageName]

  ### Read in images for figure
  DImage = util.ReadImg(DImageName)
  MImage = util.ReadImg(MImageName)
  PImage = util.ReadImg(PImageName)
  images = [DImage, MImage, PImage]

  # BE SURE TO UPDATE TESTMF WITH OPTIMIZED PARAMS
  Dimg = giveMarkedMyocyte(testImage=DImageName,tag='fig5_D',writeImage=True)
  Mimg = giveMarkedMyocyte(testImage=MImageName,tag='fig5_M',writeImage=True)
  Pimg = giveMarkedMyocyte(testImage=PImageName,tag='fig5_P',writeImage=True)

  results = [Dimg, Mimg, Pimg]
  keys = ['Distal', 'Medial', 'Proximal']
  areas = {}

  ttResults = []
  ltResults = []
  lossResults = []

  ### report responses for each case
  for i,img in enumerate(results):
    ### assess content based on cell area
    wtContent, ltContent, lossContent = assessContent(img,imgNames[i])
    ### store in lists
    ttResults.append(wtContent)
    ltResults.append(ltContent)
    lossResults.append(lossContent)

  ### generating figure
  width = 0.25
  colors = ["blue","green","red"]
  marks = ["WT","LT","Loss"]

  # opting to make a single bar chart
  N = 3
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, ttResults, width, color=colors[0])
  rects2 = ax.bar(indices+width, ltResults, width, color=colors[1])
  rects3 = ax.bar(indices+2*width, lossResults,width, color=colors[2])
  ax.set_ylabel('Normalized Content')
  ax.set_xticks(indices + width* 3/2)
  ax.set_xticklabels(keys)
  ax.legend(marks)
  plt.gcf().savefig('fig5_BarChart.pdf',dpi=300)
  plt.close()

  plt.figure()
  plt.imshow(switchBRChannels(util.markMaskOnMyocyte(DImage,DImageName)))
  plt.gcf().savefig("fig5_Raw_D.pdf",dpi=300)

  plt.figure()
  plt.imshow(switchBRChannels(util.markMaskOnMyocyte(MImage,MImageName)))
  plt.gcf().savefig("fig5_Raw_M.pdf",dpi=300)

  plt.figure()
  plt.imshow(switchBRChannels(util.markMaskOnMyocyte(PImage,PImageName)))
  plt.gcf().savefig("fig5_Raw_P.pdf",dpi=300)

def tissueComparison(fullAnalysis=True):
  '''
  Tissue level images for comparison between "Distal" region and "Proximal" region
  '''
  if fullAnalysis:
    fileTag = "tissueComparison_fullAnalysis"
  else:
    fileTag = "tissueComparison"

  ### Setup cases for use
  filterTwoSarcomereSize = 25
  cases = dict()
  
  cases['WTLike'] = empty()
  cases['WTLike'].loc_um = [3694,0]
  cases['WTLike'].extent_um = [300,300]
  
  cases['MILike'] = empty()
  cases['MILike'].loc_um = [2100,3020]
  cases['MILike'].extent_um = [300,300]

  tissue.SetupCase(cases['WTLike'])
  tissue.SetupCase(cases['MILike'])

  ### Store subregions for later marking
  cases['WTLike'].subregionOrig = cases['WTLike'].subregion.copy()
  cases['MILike'].subregionOrig = cases['MILike'].subregion.copy()

  ### Preprocess and analyze tissue subsections
  cases['WTLike'] = analyzeTissueCase(cases['WTLike'])
  cases['MILike'] = analyzeTissueCase(cases['MILike'])

  ### Save hits from the analysis
  displayTissueCaseHits(cases['WTLike'],tag=fileTag+'_Distal')
  displayTissueCaseHits(cases['MILike'],tag=fileTag+'_Proximal')

  if fullAnalysis:
    ### Quantify TT content per square micron
    cases['WTLike'].area = float(cases['WTLike'].extent_um[0] * cases['WTLike'].extent_um[1])
    cases['MILike'].area = float(cases['MILike'].extent_um[0] * cases['MILike'].extent_um[1])
    cases['WTLike'].TTcontent = float(np.sum(cases['WTLike'].pasted)) / cases['WTLike'].area
    cases['MILike'].TTcontent = float(np.sum(cases['MILike'].pasted)) / cases['MILike'].area
    ## Normalize TT content since it's fairly arbitrary
    cases['MILike'].TTcontent /= cases['WTLike'].TTcontent
    cases['WTLike'].TTcontent /= cases['WTLike'].TTcontent

    print "WT TT Content:", cases['WTLike'].TTcontent
    print "MI TT Content:", cases['MILike'].TTcontent

    ### Quantify TA 
    cases['WTLike'].TAcontent = float(np.sum(cases['WTLike'].TApasted))
    cases['MILike'].TAcontent = float(np.sum(cases['MILike'].TApasted))
    ### Normalize
    cases['MILike'].TAcontent /= cases['WTLike'].TAcontent
    cases['WTLike'].TAcontent /= cases['WTLike'].TAcontent

    print "WT TA Content:", cases['WTLike'].TAcontent
    print "MI TA Content:", cases['MILike'].TAcontent


  ### Make Bar Chart of TT content
  #width = 0.75
  #N = 2
  #indices = np.arange(N) + width
  #fig,ax = plt.subplots()
  #rects1 = ax.bar(indices[0], cases['WTLike'].TTcontent, width, color='blue',align='center')
  #rects2 = ax.bar(indices[1], cases['MILike'].TTcontent, width, color='red',align='center')
  #ax.set_ylabel('Normalized TT Content to WT',fontsize=24)
  #plt.sca(ax)
  #plt.xticks(indices,['Conserved','Perturbed'],fontsize=24)
  #ax.set_ylim([0,1.2])
  #plt.gcf().savefig(fileTag+'_TTcontent.pdf',dpi=300)

  ### Save enhanced original images for figure 
  plt.figure()
  plt.imshow(cases['WTLike'].displayImg,cmap='gray',vmin=0,vmax=255)
  plt.axis('off')
  plt.gcf().savefig(fileTag+'_Distal_enhancedImg.pdf',dpi=600)

  plt.figure()
  plt.imshow(cases['MILike'].displayImg,cmap='gray',vmin=0,vmax=255)
  plt.axis('off')
  plt.gcf().savefig(fileTag+'_Proximal_enhancedImg.pdf',dpi=600)

  ### Find angle counts for each rotation
  #cases['WTLike'].results.stackedAngles = cases['WTLike'].results.stackedAngles[np.where(
  #                                        cases['WTLike'].results.stackedAngles != -1
  #                                        )]
  #cases['MILike'].results.stackedAngles = cases['MILike'].results.stackedAngles[np.where(
  #                                        cases['MILike'].results.stackedAngles != -1
  #                                        )]


  ### Write stacked angles histogram
  #giveAngleHistogram(cases['WTLike'].results.stackedAngles,cases['WTLike'].iters,fileTag+"_Distal")
  #giveAngleHistogram(cases['MILike'].results.stackedAngles,cases['MILike'].iters,fileTag+"_Proximal")

def figAngleHeterogeneity():
  '''
  Figure to showcase the heterogeneity of striation angle present within the tissue
    sample
  NOTE: Not used in paper
  '''
  ### Setup case for use
  filterTwoSarcomereSize = 25
  case = empty()
  case.loc_um = [2800,3250]
  case.extent_um = [300,300]

  tissue.SetupCase(case)

  ### Store subregions for later marking
  case.subregionOrig = case.subregion.copy()

  ### Preprocess and analyze tissue subsections
  case = analyzeTissueCase(case)

  ### Save hits from the analysis
  displayTissueCaseHits(case,tag='figHeterogeneousAngles')

  ### Quantify TT content per square micron
  case.area = float(case.extent_um[0] * case.extent_um[1])
  case.TTcontent = float(np.sum(case.pasted)) / case.area
  ## Normalize TT content since it's fairly arbitrary
  case.TTcontent /= case.TTcontent

  plt.figure()
  plt.imshow(case.results.stackedAngles)
  plt.colorbar()
  plt.show()

  print case.results.stackedAngles

def full_ROC():
  '''
  Routine to generate the necessary ROC figures based on hand-annotated images
  '''

  root = "./myoimages/"

  imgNames = {'HF':"HF_annotation_testImg.png",
              'MI':"MI_annotation_testImg.png",
              'Control':"Sham_annotation_testImg.png"
              }

  # images that have hand annotation marked
  annotatedImgNames = {'HF':'HF_annotation_trueImg.png',
                       'Control':'Sham_annotation_trueImg.png',
                       'MI':'MI_annotation_trueImg.png'
                       }

  for key,imgName in imgNames.iteritems():
      print imgName
      ### setup dataset
      dataSet = optimizer.DataSet(
                  root = root,
                  filter1TestName = root + imgName,
                  filter1TestRegion=None,
                  filter1PositiveTest = root + annotatedImgNames[key],
                  pasteFilters=True
                  )

      ### run func that writes scores to hdf5 file
      myocyteROC(dataSet,key,threshes = np.linspace(0.05,0.7,30))
  
  ### read data from hdf5 files
  # make big dictionary
  bigData = {}
  bigData['MI'] = {}
  bigData['HF'] = {}
  bigData['Control'] = {}
  ### read in data from each myocyte
  for key,nestDict in bigData.iteritems():
    nestDict['WT'] = pd.read_hdf(key+"_WT.h5",'table')
    nestDict['LT'] = pd.read_hdf(key+"_LT.h5",'table')
    nestDict['Loss'] = pd.read_hdf(key+"_Loss.h5",'table')

  ### Go through and normalize all false positives and true positives
  for key,nestDict in bigData.iteritems():
    for key2,nestDict2 in nestDict.iteritems():
      #print key
      #print key2
      nestDict2['filter1PS'] /= np.max(nestDict2['filter1PS'])
      nestDict2['filter1NS'] /= np.max(nestDict2['filter1NS'])

  ### Figure generation
  plt.rcParams.update(plt.rcParamsDefault)
  f, axs = plt.subplots(3,2,figsize=[7,12])
  plt.subplots_adjust(wspace=0.5,bottom=0.05,top=0.95,hspace=0.25)
  locDict = {'Control':0,'MI':1,'HF':2}
  for key,loc in locDict.iteritems():
    ### writing detection rate fig
    axs[loc,0].scatter(bigData[key]['WT']['filter1Thresh'], 
                     bigData[key]['WT']['filter1PS'],label='WT',c='b')
    axs[loc,0].scatter(bigData[key]['LT']['filter1Thresh'], 
                     bigData[key]['LT']['filter1PS'],label='LT',c='g')
    axs[loc,0].scatter(bigData[key]['Loss']['filter1Thresh'],
                     bigData[key]['Loss']['filter1PS'],label='Loss',c='r')
    axs[loc,0].set_title(key+" Detection Rate",size=12)
    axs[loc,0].set_xlabel('Threshold')
    axs[loc,0].set_ylabel('Detection Rate')
    axs[loc,0].set_ylim([0,1])
    axs[loc,0].set_xlim(xmin=0)
  
    ### writing ROC fig
    axs[loc,1].set_title(key+" ROC",size=12)
    axs[loc,1].scatter(bigData[key]['WT']['filter1NS'], 
                       bigData[key]['WT']['filter1PS'],label='WT',c='b')
    axs[loc,1].scatter(bigData[key]['LT']['filter1NS'],    
                       bigData[key]['LT']['filter1PS'],label='LT',c='g')
    axs[loc,1].scatter(bigData[key]['Loss']['filter1NS'],    
                       bigData[key]['Loss']['filter1PS'],label='Loss',c='r')
    ### giving 50% line
    vert = np.linspace(0,1,10)
    axs[loc,1].plot(vert,vert,'k--')

    axs[loc,1].set_xlim([0,1])
    axs[loc,1].set_ylim([0,1])
    axs[loc,1].set_xlabel('False Positive Rate (Normalized)')
    axs[loc,1].set_ylabel('True Positive Rate (Normalized)')

  plt.gcf().savefig('figS1.pdf',dpi=300)

def tissueBloodVessel():
  '''
  Routine to generate the figure showcasing heterogeneity of striation angle
    in the tissue sample
  '''
  fileTag = 'tissueBloodVessel'

  ### setup case
  case = empty()
  case.loc_um = [2477,179]
  case.extent_um = [350,350]
  case.orig = tissue.Setup()
  case.subregion = tissue.get_fiji(case.orig,case.loc_um,case.extent_um)
  #case = tissue.SetupTest()

  ### Store subregions for later marking
  case.subregionOrig = case.subregion.copy()

  ### Preprocess and analyze the case
  case = analyzeTissueCase(case)

  ### display the hits of the case
  displayTissueCaseHits(case,fileTag)

  ### Save enhanced original images for figure
  plt.figure()
  plt.imshow(case.displayImg,cmap='gray',vmin=0,vmax=255)
  plt.gcf().savefig(fileTag+'_enhancedImg.pdf',dpi=300)

def algComparison():
  '''
  Routine to compare the GPU and CPU implementation of code
  '''

  fileTag = "algComparison"

  ### setup case
  caseGPU = empty()
  caseGPU.loc_um = [2477,179]
  caseGPU.extent_um = [350,350]
  caseGPU.orig = tissue.Setup()
  caseGPU.subregion = tissue.get_fiji(caseGPU.orig,caseGPU.loc_um,caseGPU.extent_um)

  ### Store subregions for later marking
  caseGPU.subregionOrig = caseGPU.subregion.copy()

  ### Preprocess 
  caseGPU = preprocessTissueCase(caseGPU)

  ### make necessary copies for CPU case
  caseCPU = empty()
  caseCPU.loc_um = [2477,179]
  caseCPU.extent_um = [350,350]
  caseCPU.orig = caseGPU.orig.copy()
  caseCPU.degreesOffCenter = caseGPU.degreesOffCenter
  caseCPU.subregionOrig = caseGPU.subregionOrig.copy()
  caseCPU.subregion = caseGPU.subregion.copy()
  caseCPU.displayImg = caseGPU.displayImg.copy()
  
  ### analyze the case
  caseGPU = analyzeTissueCase(caseGPU,preprocess=False)

  ### display the hits of the case
  displayTissueCaseHits(caseGPU,fileTag+"_GPU")

  ### store results
  GPUpastedHits = caseGPU.pasted

  ### Preprocess and analyze the case
  caseCPU = analyzeTissueCase(caseCPU,preprocess=False,useGPU=False)

  ### display the hits of the case
  displayTissueCaseHits(caseCPU,fileTag+"_CPU")

  ### store results
  CPUpastedHits = caseCPU.pasted

  ### save original enhanced image for comparison
  plt.figure()
  plt.imshow(caseGPU.displayImg,cmap='gray',vmin=0,vmax=255)
  plt.gcf().savefig(fileTag+'_enhancedImg.pdf',dpi=600)
  
  ### do comparison between GPU and CPU results
  comparison = np.abs(GPUpastedHits - CPUpastedHits).astype(np.float32)
  comparison /= np.max(comparison)
  plt.figure()
  plt.imshow(comparison,cmap='gray')
  plt.colorbar()
  plt.gcf().savefig(fileTag+'_comparison.pdf',dpi=600)

def YAML_example():
  '''
  Routine to generate the example YAML output in the supplement
  '''
  detect.updatedSimpleYaml("ex.yml")
  
def shaveFig(fileName,padY=None,padX=None,whiteSpace=None):
  '''
  Aggravating way of shaving a figure's white space down to an acceptable level
    and adding in enough whitespace to label the figure
  '''

  img = util.ReadImg(fileName,cvtColor=False)
  imgDims = np.shape(img)

  ### get sum along axis
  rowSum = np.sum(img[:,:,0],axis=1).astype(np.float32)
  colSum = np.sum(img[:,:,0],axis=0).astype(np.float32)

  ### get average along axis
  rowAvg = rowSum / float(imgDims[1])
  colAvg = colSum / float(imgDims[0])

  ### find where the first occurrence of non-white space is
  firstNonWhiteRowIdx = np.argmax((rowAvg-255.)**2. != 0)
  firstNonWhiteColIdx = np.argmax((colAvg-255.)**2. != 0)
  invNonWhiteRowIdx = np.argmax((rowAvg[::-1]-255.)**2. != 0)
  invNonWhiteColIdx = np.argmax((colAvg[::-1]-255.)**2. != 0)

  ### add some padding in
  if padX == None:
    padX = 20
  if padY == None:
    padY = 60 
  firstNonWhiteRowIdx -= padY
  firstNonWhiteColIdx -= padX
  invNonWhiteRowIdx -= padY
  invNonWhiteColIdx -= padX

  idxs = [firstNonWhiteRowIdx, invNonWhiteRowIdx, firstNonWhiteColIdx, invNonWhiteColIdx]
  for i,idx in enumerate(idxs):
    if idx <= 0:
      idxs[i] = 1

  if whiteSpace == None:
    extraWhiteSpace = 100
  else:
    extraWhiteSpace = whiteSpace
  
  newImg = np.zeros((imgDims[0]-idxs[0]-idxs[1]+extraWhiteSpace,
                     imgDims[1]-idxs[2]-idxs[3],3),
                    dtype=np.uint8)

  for channel in range(3):
    newImg[extraWhiteSpace:,:,channel] = img[idxs[0]:-idxs[1],
                                             idxs[2]:-idxs[3],channel]
    newImg[:extraWhiteSpace,:,channel] = 255
  
  cv2.imwrite(fileName,newImg)

def preprocessTissueCase(case):
  ### Preprocess subregions
  case.subregion, case.degreesOffCenter = pp.reorient(
          case.subregion
          )

  ### save image for display later
  ## I'm considering writing a better routine to enhance the image for figure quality but not necessarily for algorithm quality
  brightnessDamper = 0.6
  case.displayImg = case.subregion.copy().astype(np.float32) / float(np.max(case.subregion))
  case.displayImg *= brightnessDamper * 255.
  case.displayImg = case.displayImg.astype(np.uint8)

  return case

def analyzeTissueCase(case,
                      preprocess=True,
                      useGPU=True,
                      analyzeTA=True):
  '''
  Refactored method to analyze tissue cases 
  '''
  if preprocess:
    case = preprocessTissueCase(case)

  ### Setup Filters
  root = "./myoimages/"
  ttFilterName = root+"newSimpleWTFilter.png"
  ttPunishmentFilterName = root+"newSimpleWTPunishmentFilter.png"
  case.iters = [-25,-20,-15,-10-5,0,5,10,15,20,25]
  returnAngles = True
  ttFilter = util.LoadFilter(ttFilterName)
  ttPunishmentFilter = util.LoadFilter(ttPunishmentFilterName)

  ### Setup parameter dictionaries
  case.params = optimizer.ParamDict(typeDict="WT")
  case.params['covarianceMatrix'] = np.ones_like(case.subregion)
  case.params['mfPunishment'] = ttPunishmentFilter
  case.params['useGPU'] = useGPU

  ### Setup input classes
  case.inputs = empty()
  case.inputs.imgOrig = case.subregion.astype(np.float32) / float(np.max(case.subregion))
  case.inputs.mfOrig = ttFilter

  ### Perform filtering for TT detection
  case.results = bD.DetectFilter(case.inputs,
                                 case.params,
                                 case.iters
                                 )

  ### Modify case to perform TA detection
  if analyzeTA:
    case.TAinputs = empty()
    case.TAinputs.imgOrig = case.subregion
    case.TAinputs.displayImg = case.displayImg
    lossFilterName = root+"LossFilter.png"
    case.TAIters = [-45,0]
    lossFilter = util.LoadFilter(lossFilterName)
    case.TAinputs.mfOrig = lossFilter
    case.TAparams = optimizer.ParamDict(typeDict='Loss')

    ### Perform filtering for TA detection
    case.TAresults = bD.DetectFilter(case.TAinputs,
                                     case.TAparams,
                                     case.TAIters)

  return case

def displayTissueCaseHits(case,
                          tag,
                          displayTT=True,
                          displayTA=True):
  '''
  Displays the 'hits' returned from analyzeTissueCase() function
  '''
  ### Convert subregion back into cv2 readable format
  case.subregion = np.asarray(case.subregion * 255.,dtype=np.uint8)

  ### Mark where the filter responded and display on the images
  ## find filter dimensions
  TTy,TTx = util.measureFilterDimensions(case.inputs.mfOrig)

  ## mark unit cells on the image where the filters responded
  case.pasted = painter.doLabel(case.results,dx=TTx,dy=TTy,
                                thresh=case.params['snrThresh'])

  ## convert pasted filter image to cv2 friendly format and normalize original subregion
  case.pasted = np.asarray(case.pasted 
                           / np.max(case.pasted) 
                           * 255.,
                           dtype=np.uint8)

  ## rotate images back to the original orientation
  case.pasted = imutils.rotate(case.pasted,case.degreesOffCenter)
  case.displayImg = imutils.rotate(case.displayImg,case.degreesOffCenter)

  debug = False
  if debug:
    plt.figure()
    plt.imshow(case.displayImg,cmap='gray')
    plt.show()

  ## cut image back down to original size to get rid of borders
  imgDims = np.shape(case.subregionOrig)
  origY,origX = float(imgDims[0]), float(imgDims[1])
  newImgDims = np.shape(case.displayImg)
  newY,newX = float(newImgDims[0]),float(newImgDims[1])
  padY,padX = int((newY - origY)/2.), int((newX - origX)/2.)
  case.pasted = case.pasted[padY:-padY,
                            padX:-padX]
  case.displayImg = case.displayImg[padY:-padY,
                                    padX:-padX]

  ### Create colored image for display
  coloredImage = np.asarray((case.displayImg.copy(),
                             case.displayImg.copy(),
                             case.displayImg.copy()))
  coloredImage = np.rollaxis(coloredImage,0,start=3)
  TTchannel = 2

  ### Mark channel hits on image
  if displayTT:
    coloredImage[case.pasted != 0,TTchannel] = 255

  ### Do the same thing for the TA case
  if displayTA:
    TAy,TAx = util.measureFilterDimensions(case.TAinputs.mfOrig)
    case.TApasted = painter.doLabel(case.TAresults,dx=TAx,dy=TAy,
                                    thresh=case.TAparams['snrThresh'])
    case.TApasted = np.asarray(case.TApasted
                               / np.max(case.TApasted)
                               * 255.,
                               dtype=np.uint8)
    case.TApasted = imutils.rotate(case.TApasted,case.degreesOffCenter)
    case.TApasted = case.TApasted[padY:-padY,
                                  padX:-padX]
    TAchannel = 0
    coloredImage[case.TApasted != 0,TAchannel] = 255

  ### Plot figure and save
  plt.figure()
  plt.imshow(coloredImage,vmin=0,vmax=255)
  plt.gcf().savefig(tag+"_hits.pdf",dpi=600)

def saveWorkflowFig():
  '''
  Function that will save the images used for the workflow figure in the paper.
  Note: This is slightly subject to how one preprocesses the MI_D_73.png image.
        A slight change in angle or what subsection was selected in the preprocessing
        could slightly change how the images appear.
  '''

  imgName = "./myoimages/MI_D_73_processed.png" 
  iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
  
  colorImg,colorAngles,angleCounts = giveMarkedMyocyte(testImage=imgName,
                                                       tag="WorkflowFig",
                                                       iters=iters,
                                                       returnAngles=True,
                                                       writeImage=True)

  ### save the correlation planes
  lossFilter = util.LoadFilter("./myoimages/LossFilter.png")
  ltFilter = util.LoadFilter("./myoimages/LongitudinalFilter.png")
  wtFilter = util.LoadFilter("./myoimages/newSimpleWTFilter.png")

  origImg = util.ReadImg(imgName,cvtColor=False)
  origImg = cv2.cvtColor(origImg,cv2.COLOR_BGR2GRAY)
  lossCorr = mF.matchedFilter(origImg, lossFilter,demean=False)
  ltCorr = mF.matchedFilter(origImg, ltFilter,demean=False)
  wtCorr = mF.matchedFilter(origImg, wtFilter,demean=False)

  cropImgs = False
  if cropImgs:
    angle_output = util.ReadImg("WorkflowFig_angles_output.png",cvtColor=False)
    output = util.ReadImg("WorkflowFig_output.png",cvtColor=False)
    imgs = {"WorkflowFig_angles_output.png":angle_output, 
            "WorkflowFig_output.png":output, 
            "WorkflowFig_orig.png":origImg}

    left = 204; right = 304; top = 74; bottom = 151
    for name,img in imgs.iteritems():
      if cropImgs:
        holder = np.zeros((bottom-top,right-left,3),dtype=np.uint8)
        for channel in range(3):
          ### crop images
          holder[:,:,channel] = img[top:bottom,left:right,channel]
      cv2.imwrite(name,holder)
    lossCorr = lossCorr[top:bottom,left:right]
    ltCorr = ltCorr[top:bottom,left:right]
    wtCorr = wtCorr[top:bottom,left:right]

  plt.figure()
  plt.imshow(lossCorr,cmap='gray')
  plt.gcf().savefig("WorkflowFig_lossCorr.pdf")

  plt.figure()
  plt.imshow(ltCorr,cmap='gray')
  plt.gcf().savefig("WorkflowFig_ltCorr.pdf")

  plt.figure()
  plt.imshow(wtCorr,cmap='gray')
  plt.gcf().savefig("WorkflowFig_wtCorr.pdf")

  ### Assess content
  wtC,ltC,ldC = assessContent(colorImg,imgName=imgName)

  ### make a bar chart using routine
  content = [wtC,ltC,ldC]
  contentDict = {imgName:content}
  giveBarChartfromDict(contentDict,"WorkflowFig_Content")

  ### Make a histogram for the angles
  giveAngleHistogram(angleCounts,iters,"WorkflowFig")

def giveAngleHistogram(angleCounts,iters,tag):
  ### Make a histogram for the angles
  iters = np.asarray(iters,dtype='float')
  binSpace = iters[-1] - iters[-2]
  myBins = iters - binSpace / 2.
  myBins= np.append(myBins,myBins[-1] + binSpace)
  plt.figure()
  n, bins, patches = plt.hist(angleCounts, bins=myBins,
                              normed=True,
                              align='mid',
                              facecolor='green', alpha=0.5)
  plt.xlabel('Rotation Angle')
  plt.ylabel('Probability')
  plt.gcf().savefig(tag+"_angle_histogram.pdf",dpi=300)
  plt.close()

def giveAvgStdofDicts(ShamDict,HFDict,MI_DDict,MI_MDict,MI_PDict):
  ### make a big dictionary to iterate through
  results = {'Sham':ShamDict,'HF':HFDict,'MI_D':MI_DDict,'MI_M':MI_MDict,'MI_P':MI_PDict}
  ### make dictionaries to store results
  avgDict = {}; stdDict = {}
  for model,dictionary in results.iteritems():
    ### make holders to store results
    angleAvgs = []; angleStds = []
    for name,angleCounts in dictionary.iteritems():
      print name
      if 'angle' not in name:
        continue
      angleAvgs.append(np.mean(angleCounts))
      angleStds.append(np.std(angleCounts))
      print "Average striation angle:",angleAvgs[-1]
      print "Standard deviation of striation angle:",angleStds[-1]
    avgDict[model] = angleAvgs
    stdDict[model] = angleStds

  ### Normalize Standard Deviations to Sham Standard Deviation
  ShamAvgStd = np.mean(stdDict['Sham'])
  stdStdDev = {}
  for name,standDev in stdDict.iteritems():
    standDev = np.asarray(standDev,dtype=float) / ShamAvgStd
    stdDict[name] = np.mean(standDev)
    stdStdDev[name] = np.std(standDev)

  ### Make bar chart for angles 
  # need to have results in ordered arrays...
  names = ['Sham', 'HF', 'MI_D', 'MI_M', 'MI_P']
  avgs = []; stds = []
  for name in names:
    avgs.append(avgDict[name])
    stds.append(stdDict[name])
  width = 0.25
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  for i,name in enumerate(names):
    ax.bar(indices+i*width, stdDict[name], width, yerr=stdStdDev[name],ecolor='k',alpha=0.5)
  ax.set_ylabel("Average Angle Standard Deviation Normalized to Sham")
  xtickLocations = np.arange(len(names)) * width + width*3./2.
  ax.set_xticks(xtickLocations)
  ax.set_xticklabels(names,rotation='vertical')
  plt.gcf().savefig("Whole_Dataset_Angles.pdf",dpi=300)

def analyzeAllMyo(root="/net/share/dfco222/data/TT/LouchData/processedWithIntelligentThresholding/"):
  '''
  Function to iterate through a directory containing images that have already
  been preprocessed by preprocessing.py
  This directory can contain masks but it is not necessary
  '''
  ### instantiate dicitionary to hold content values
  Sham = {}; MI_D = {}; MI_M = {}; MI_P = {}; HF = {};

  for name in os.listdir(root):
    if "mask" in name:
      continue
    print name
    iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
    ### iterate through names and mark the images
    markedMyocyte,_,angleCounts = giveMarkedMyocyte(testImage=root+name,
                                                    tag=name[:-4],
                                                    iters=iters,
                                                    writeImage=True,
                                                    returnAngles=True)
    ### save raw image with ROI marked
    cImg = util.ReadImg(root+name)
    cImg = util.markMaskOnMyocyte(cImg,root+name)
    plt.figure()
    plt.imshow(switchBRChannels(cImg))
    plt.gcf().savefig(name[:-4]+'_markedMask.pdf')
    plt.close()

    ### hacky way to get percent of hits within range of 5 degrees from minor axis
    idxs = [4,5,6]
    totalHits = len(angleCounts)
    angleCountsNP = np.asarray(angleCounts)
    hitsInRange =   np.count_nonzero(np.equal(angleCounts, iters[idxs[0]])) \
                  + np.count_nonzero(np.equal(angleCounts, iters[idxs[1]])) \
                  + np.count_nonzero(np.equal(angleCounts, iters[idxs[2]])) 
    print "Percentage of WT hits within 5 degrees of minor axis:", float(hitsInRange)/float(totalHits) * 100.

    ### assess content
    wtC, ltC, lossC = assessContent(markedMyocyte,imgName=root+name)
    content = np.asarray([wtC, ltC, lossC],dtype=float)

    ### store content in respective dictionary
    if 'Sham' in name:
      Sham[name] = content
      Sham[name+'_angles'] = angleCounts
    elif 'HF' in name:
      HF[name] = content
      HF[name+'_angles'] = angleCounts
    elif 'MI' in name:
      if '_D' in name:
        MI_D[name] = content
        MI_D[name+'_angles'] = angleCounts
      elif '_M' in name:
        MI_M[name] = content
        MI_M[name+'_angles'] = angleCounts
      elif '_P' in name:
        MI_P[name] = content
        MI_P[name+'_angles'] = angleCounts

    ### make angle histogram for the data
    giveAngleHistogram(angleCounts,iters,tag=name[:-4])

  ### use function to construct and write bar charts for each content dictionary
  giveBarChartfromDict(Sham,'Sham')
  giveBarChartfromDict(HF,'HF')
  giveMIBarChart(MI_D,MI_M,MI_P)
  giveAvgStdofDicts(Sham,HF,MI_D,MI_M,MI_P)

def analyzeSingleMyo(name,twoSarcSize):
   realName = name#+"_processed.png"
   iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
   markedMyocyte,_,angleCounts = giveMarkedMyocyte(testImage=realName,
                                     ImgTwoSarcSize=twoSarcSize,
                                     tag=name,
                                     writeImage=True,
                                     returnAngles=True)
   ### assess content
   wtC, ltC, lossC = assessContent(markedMyocyte,imgName=realName)
   #content = np.asarray([wtC, ltC, lossC],dtype=float)
   #content /= np.max(content)

   ### hacky way to get percent of hits within range of 5 degrees from minor axis
   idxs = [4,5,6]
   totalHits = len(angleCounts)
   angleCountsNP = np.asarray(angleCounts)
   hitsInRange =   np.count_nonzero(np.equal(angleCounts, iters[idxs[0]])) \
                 + np.count_nonzero(np.equal(angleCounts, iters[idxs[1]])) \
                 + np.count_nonzero(np.equal(angleCounts, iters[idxs[2]]))
   print "Percentage of WT hits within 5 degrees of minor axis:", float(hitsInRange)/float(totalHits) * 100.

def giveBarChartfromDict(dictionary,tag):
  ### instantiate lists to contain contents
  wtC = []; ltC = []; lossC = [];
  for name,content in dictionary.iteritems():
    if "angle" in name:
      continue
    wtC.append(content[0])
    ltC.append(content[1])
    lossC.append(content[2])

  wtC = np.asarray(wtC)
  ltC = np.asarray(ltC)
  lossC = np.asarray(lossC)

  wtAvg = np.mean(wtC)
  ltAvg = np.mean(ltC)
  lossAvg = np.mean(lossC)

  wtStd = np.std(wtC)
  ltStd = np.std(ltC)
  lossStd = np.std(lossC)

  ### now make a bar chart from this
  colors = ["blue","green","red"]
  marks = ["WT", "LT", "Loss"]
  width = 0.25
  N = 1
  indices = np.arange(N) + width
  fig,ax = plt.subplots()
  rects1 = ax.bar(indices, wtAvg, width, color=colors[0],yerr=wtStd,ecolor='k')
  rects2 = ax.bar(indices+width, ltAvg, width, color=colors[1],yerr=ltStd,ecolor='k')
  rects3 = ax.bar(indices+2*width, lossAvg, width, color=colors[2],yerr=lossStd,ecolor='k')
  ax.set_ylabel('Normalized Content')
  ax.legend(marks)
  ax.set_xticks([])
  ax.set_ylim([0,1])
  plt.gcf().savefig(tag+'_BarChart.pdf',dpi=300)

def giveMIBarChart(MI_D, MI_M, MI_P):
  '''
  Gives combined bar chart for all three proximities to the infarct.
  MI_D, MI_M, and MI_P are all dictionaries with structure:
    dict['file name'] = [wtContent, ltContent, lossContent]
  where the contents are floats
  '''

  wtAvgs = {}; wtStds = {}; ltAvgs = {}; ltStds = {}; lossAvgs = {}; lossStds = {};

  DwtC = []; DltC = []; DlossC = [];
  for name, content in MI_D.iteritems():
    if 'angle' in name:
      continue
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    DwtC.append(wtC)
    DltC.append(ltC)
    DlossC.append(lossC)
  wtAvgs['D'] = np.mean(DwtC)
  wtStds['D'] = np.std(DwtC)
  ltAvgs['D'] = np.mean(DltC)
  ltStds['D'] = np.std(DltC)
  lossAvgs['D'] = np.mean(DlossC)
  lossStds['D'] = np.std(DlossC)

  MwtC = []; MltC = []; MlossC = [];
  for name, content in MI_M.iteritems():
    if 'angle' in name:
      continue
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    MwtC.append(wtC)
    MltC.append(ltC)
    MlossC.append(lossC)
  wtAvgs['M'] = np.mean(MwtC)
  wtStds['M'] = np.std(MwtC)
  ltAvgs['M'] = np.mean(MltC)
  ltStds['M'] = np.std(MltC)
  lossAvgs['M'] = np.mean(MlossC)
  lossStds['M'] = np.std(MlossC)


  PwtC = []; PltC = []; PlossC = [];
  for name, content in MI_P.iteritems():
    if 'angle' in name:
      continue
    wtC = content[0]
    ltC = content[1]
    lossC = content[2]
    PwtC.append(wtC)
    PltC.append(ltC)
    PlossC.append(lossC)
  wtAvgs['P'] = np.mean(PwtC)
  wtStds['P'] = np.std(PwtC)
  ltAvgs['P'] = np.mean(PltC)
  ltStds['P'] = np.std(PltC)
  lossAvgs['P'] = np.mean(PlossC)
  lossStds['P'] = np.std(PlossC)

  colors = ["blue","green","red"]
  marks = ["WT", "LT", "Loss"]
  width = 1.0
  N = 11
  indices = np.arange(N)*width + width/4.
  fig,ax = plt.subplots()

  ### plot WT
  rects1 = ax.bar(indices[0], wtAvgs['D'], width, color=colors[0],yerr=wtStds['D'],ecolor='k',label='WT')
  rects2 = ax.bar(indices[1], wtAvgs['M'], width, color=colors[0],yerr=wtStds['M'],ecolor='k',label='WT')
  rects3 = ax.bar(indices[2], wtAvgs['P'], width, color=colors[0],yerr=wtStds['P'],ecolor='k',label='WT')

  ### plot LT
  rects4 = ax.bar(indices[4], ltAvgs['D'], width, color=colors[1],yerr=ltStds['D'],ecolor='k',label='LT')
  rects5 = ax.bar(indices[5], ltAvgs['M'], width, color=colors[1],yerr=ltStds['M'],ecolor='k',label='LT')
  rects6 = ax.bar(indices[6], ltAvgs['P'], width, color=colors[1],yerr=ltStds['P'],ecolor='k',label='LT')

  ### plot Loss
  rects7 = ax.bar(indices[8], lossAvgs['D'], width, color=colors[2],yerr=lossStds['D'],ecolor='k',label='Loss')
  rects8 = ax.bar(indices[9], lossAvgs['M'], width, color=colors[2],yerr=lossStds['M'],ecolor='k',label='Loss')
  rects9 = ax.bar(indices[10],lossAvgs['P'], width, color=colors[2],yerr=lossStds['P'],ecolor='k',label='Loss')

  ax.set_ylabel('Normalized Content')
  ax.legend(handles=[rects1,rects4,rects7])
  newInd = indices + width/2.
  ax.set_xticks(newInd)
  ax.set_xticklabels(['D', 'M','P','','D','M','P','','D','M','P'])
  ax.set_ylim([0,1])
  plt.gcf().savefig('MI_BarChart.pdf',dpi=300)

def markPastedFilters(
      lossMasked, ltMasked, wtMasked, cI,
      lossName="./myoimages/LossFilter.png",
      ltName="./myoimages/LongitudinalFilter.png",
      wtName="./myoimages/newSimpleWTFilter.png"
      ):
  '''
  Given masked stacked hits for the 3 filters and a doctored colored image, 
  function will paste filter sized boxes around the characterized regions
  and return the colored image with filter sized regions colored.

  NOTE: Colored image was read in (not grayscale) and 1 was subtracted from
  the image. This was necessary for the thresholding to work with the painter
  function
  '''
  # exploiting architecture of painter function to mark hits for me
  Lossholder = empty()
  Lossholder.stackedHits = lossMasked
  LTholder = empty()
  LTholder.stackedHits = ltMasked
  WTholder = empty()
  WTholder.stackedHits = wtMasked

  ### load in filters to get filter dimensions
  lossFilt = util.LoadFilter(lossName)
  ltFilt = util.LoadFilter(ltName)
  wtFilt = util.LoadFilter(wtName)

  ### get filter dimensions
  lossy,lossx = util.measureFilterDimensions(lossFilt)
  LTy, LTx = util.measureFilterDimensions(ltFilt)
  WTy, WTx = util.measureFilterDimensions(wtFilt)

  ### we want to mark WT last since that should be the most stringent
  # Opting to mark Loss, then Long, then WT
  labeledLoss = painter.doLabel(Lossholder,dx=lossx,dy=lossy,thresh=254)
  labeledLT = painter.doLabel(LTholder,dx=LTx,dy=LTy,thresh=254)
  labeledWT = painter.doLabel(WTholder,dx=WTx,dy=WTy,thresh=254)

  ### perform masking
  WTmask = labeledWT.copy()
  LTmask = labeledLT.copy()
  Lossmask = labeledLoss.copy()

  WTmask[labeledLoss] = False
  WTmask[labeledLT] = False
  LTmask[labeledLoss] = False
  LTmask[WTmask] = False # prevents double marking of WT and LT

  alpha = 1.0
  cI[:,:,2][Lossmask] = int(round(alpha * 255))
  cI[:,:,1][LTmask] = int(round(alpha * 255))
  cI[:,:,0][WTmask] = int(round(alpha * 255))

  return cI

def WT_Filtering(inputs,
                 iters,
                 ttFilterName,
                 wtPunishFilterName,
                 ttThresh=None,
                 wtGamma=None,
                 returnAngles=False):
  '''
  Takes inputs class that contains original image and performs WT filtering on the image
  '''
  ttFilter = util.LoadFilter(ttFilterName)
  inputs.mfOrig = ttFilter
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(inputs.imgOrig)
  WTparams['mfPunishment'] = util.LoadFilter(wtPunishFilterName)
  WTparams['useGPU'] = inputs.useGPU
  if ttThresh != None:
    WTparams['snrThresh'] = ttThresh
  if wtGamma != None:
    WTparams['gamma'] = wtGamma
  print "WT Filtering"
  WTresults = bD.DetectFilter(inputs,WTparams,iters,returnAngles=returnAngles)  

  return WTresults

def LT_Filtering(inputs,
                 iters,
                 ltFilterName,
                 ltThresh=None,
                 ltStdThresh=None,
                 returnAngles=False
                 ):
  '''
  Takes inputs class that contains original image and performs LT filtering on the image
  '''

  print "LT filtering"
  inputs.mfOrig = util.LoadFilter(ltFilterName)
  LTparams = optimizer.ParamDict(typeDict='LT')
  if ltThresh != None:
    LTparams['snrThresh'] = ltThresh
  if ltStdThresh != None:
    LTparams['stdDevThresh'] = ltStdThresh
  LTparams['useGPU'] = inputs.useGPU
  LTresults = bD.DetectFilter(inputs,LTparams,iters,returnAngles=returnAngles)

  return LTresults

def Loss_Filtering(inputs,
                   lossFilterName,
                   iters=None,
                   lossThresh=None,
                   lossStdThresh=None,
                   returnAngles=False,
                   ):
  '''
  Takes inputs class that contains original image and performs Loss filtering on the image
  '''
  print "Loss filtering"
  inputs.mfOrig = util.LoadFilter(lossFilterName)
  Lossparams = optimizer.ParamDict(typeDict='Loss')
  Lossparams['useGPU'] = inputs.useGPU
  if iters != None:
    Lossiters = iters
  else:
    Lossiters = [0, 45] # don't need many rotations for loss filtering
  if lossThresh != None:
    Lossparams['snrThresh'] = lossThresh
  if lossStdThresh != None:
    Lossparams['stdDevThresh'] = lossStdThresh
  Lossresults = bD.DetectFilter(inputs,Lossparams,Lossiters,returnAngles=returnAngles)

  return Lossresults

def giveMarkedMyocyte(
      ttFilterName="./myoimages/newSimpleWTFilter.png",
      ltFilterName="./myoimages/LongitudinalFilter.png",
      lossFilterName="./myoimages/LossFilter.png",
      wtPunishFilterName="./myoimages/newSimpleWTPunishmentFilter.png",
      ltPunishFilterName="./myoimages/newLTPunishmentFilter.png",
      testImage="./myoimages/MI_D_73_annotation.png",
      ImgTwoSarcSize=None,
      tag = "default_",
      writeImage = False,
      ttThresh=None,
      ltThresh=None,
      lossThresh=None,
      wtGamma=None,
      ltStdThresh=None,
      lossStdThresh=None,
      iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
      returnAngles=False,
      returnPastedFilter=True,
      useGPU=False,
      fileExtension=".pdf"
      ):
 
  start = time.time()

  ### Read in preprocessed image
  img = util.ReadImg(testImage,renorm=True)

  ### defining inputs to be read by DetectFilter function
  inputs = empty()
  inputs.imgOrig = ReadResizeApplyMask(img,testImage,25,25) # just applies mask
  inputs.useGPU = useGPU

  ### WT filtering
  if ttFilterName != None:
    WTresults = WT_Filtering(inputs,iters,ttFilterName,wtPunishFilterName,ttThresh,wtGamma,returnAngles)
    WTstackedHits = WTresults.stackedHits
  else:
    WTstackedHits = np.zeros_like(inputs.imgOrig)

  ### LT filtering
  if ltFilterName != None:
    LTresults = LT_Filtering(inputs,iters,ltFilterName,ltThresh,ltStdThresh,returnAngles)
    LTstackedHits = LTresults.stackedHits
  else:
    LTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Loss filtering
  if lossFilterName != None:
    Lossresults = Loss_Filtering(inputs,lossFilterName,lossThresh,lossStdThresh,returnAngles)
    LossstackedHits = Lossresults.stackedHits
  else:
    LossstackedHits = np.zeros_like(inputs.imgOrig)
 
  ### Read in colored image for marking hits
  cI = util.ReadImg(testImage,cvtColor=False).astype(np.float64)
  # killing the brightness a bit so looking at results isn't like staring at the sun
  cI *= 0.85
  cI = np.round(cI).astype(np.uint8)

  ### Marking superthreshold hits for loss filter
  LossstackedHits[LossstackedHits != 0] = 255
  LossstackedHits = np.asarray(LossstackedHits, dtype='uint8')

  ### applying a loss mask to attenuate false positives from WT and Longitudinal filter
  WTstackedHits[LossstackedHits == 255] = 0
  LTstackedHits[LossstackedHits == 255] = 0

  ### marking superthreshold hits for longitudinal filter
  LTstackedHits[LTstackedHits != 0] = 255
  LTstackedHits = np.asarray(LTstackedHits, dtype='uint8')

  ### masking WT response with LT mask so there is no overlap in the markings
  WTstackedHits[LTstackedHits == 255] = 0

  ### marking superthreshold hits for WT filter
  WTstackedHits[WTstackedHits != 0] = 255
  WTstackedHits = np.asarray(WTstackedHits, dtype='uint8')

  ### apply preprocessed masks
  wtMasked = ReadResizeApplyMask(WTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  ltMasked = ReadResizeApplyMask(LTstackedHits,testImage,ImgTwoSarcSize,
                                 filterTwoSarcSize=ImgTwoSarcSize)
  lossMasked = ReadResizeApplyMask(LossstackedHits,testImage,ImgTwoSarcSize,
                                   filterTwoSarcSize=ImgTwoSarcSize)

  if not returnPastedFilter:
    ### create holders for marking channels
    WTcopy = cI[:,:,0]
    LTcopy = cI[:,:,1]
    Losscopy = cI[:,:,2]

    ### color corrresponding channels
    WTcopy[wtMasked == 255] = 255
    LTcopy[ltMasked == 255] = 255
    Losscopy[lossMasked == 255] = 255

    ### mark mask outline on myocyte
    cI = util.markMaskOnMyocyte(cI,testImage)
    if writeImage:
      ### mark mask outline on myocyte
      #cI_written = util.markMaskOnMyocyte(cI,testImage)
      cI_written = cI

      ### write output image
      plt.figure()
      plt.imshow(switchBRChannels(cI_written))
      plt.gcf().savefig(tag+"_output"+fileExtension,dpi=300)

  if returnPastedFilter:
    dummy = np.zeros_like(cI)
    dummy = markPastedFilters(lossMasked, ltMasked, wtMasked, dummy)
    ### apply mask again so as to avoid content > 1.0
    dummy = ReadResizeApplyMask(dummy,testImage,ImgTwoSarcSize,filterTwoSarcSize=ImgTwoSarcSize)
    cI[dummy==255] = 255
  
    if writeImage:
      ### mark mask outline on myocyte
      #cI_written = util.markMaskOnMyocyte(cI.copy(),testImage)
      cI_written = cI

      ### write outputs	  
      plt.figure()
      plt.imshow(switchBRChannels(cI_written))
      plt.gcf().savefig(tag+"_output"+fileExtension,dpi=300)

  if returnAngles:
    cImg = util.ReadImg(testImage,cvtColor=False)

    ### perform smoothing on the original image
    dim = 5
    kernel = np.ones((dim,dim),dtype=np.float32)
    kernel /= np.sum(kernel)
    smoothed = mF.matchedFilter(inputs.imgOrig,kernel,demean=False)

    ### make longer WT filter so more robust to striation angle deviation
    ttFilter = util.LoadFilter(ttFilterName)
    longFilter = np.concatenate((ttFilter,ttFilter,ttFilter))
    
    rotInputs = empty()
    rotInputs.imgOrig = smoothed
    rotInputs.mfOrig = longFilter

    params = optimizer.ParamDict(typeDict='WT')
    params['snrThresh'] = 0 # to pull out max hit
    params['filterMode'] = 'simple' # we want no punishment since that causes high variation
    
    ### perform simple filtering
    smoothedWTresults = bD.DetectFilter(rotInputs,params,iters,returnAngles=returnAngles)
    smoothedHits = smoothedWTresults.stackedAngles

    ### pull out actual hits from smoothed results
    smoothedHits[WTstackedHits == 0] = -1

    coloredAngles = painter.colorAngles(cImg,smoothedHits,iters)

    coloredAnglesMasked = ReadResizeApplyMask(coloredAngles,testImage,
                                              ImgTwoSarcSize,
                                              filterTwoSarcSize=ImgTwoSarcSize)
    stackedAngles = smoothedHits
    dims = np.shape(stackedAngles)
    angleCounts = []
    for i in range(dims[0]):
      for j in range(dims[1]):
        rotArg = stackedAngles[i,j]
        if rotArg != -1:
          ### indicates this is a hit
          angleCounts.append(iters[rotArg])

    if writeImage:
      #cv2.imwrite(tag+"_angles_output.png",coloredAnglesMasked)
      plt.figure()
      plt.imshow(switchBRChannels(coloredAnglesMasked))
      plt.gcf().savefig(tag+"_angles_output"+fileExtension)
    
    end = time.time()
    tElapsed = end - start
    print "Total Elapsed Time: {}s".format(tElapsed)
    return cI, coloredAnglesMasked, angleCounts

  end = time.time()
  tElapsed = end - start
  print "Total Elapsed Time: {}s".format(tElapsed)
  return cI 

def give3DMarkedMyocyte(
      testImage,
      ttFilterName=None,
      ltFilterName=None,
      taFilterName=None,
      ttPunishFilterName=None,
      ltPunishFilterName=None,
      ttThresh=None,
      ltThresh=None,
      lossThresh=None,
      wtGamma=None,
      ltStdThresh=None,
      lossStdThresh=None,
      ImgTwoSarcSize=None,
      tag = "default_",
      xiters=[-10,0,10],
      yiters=[-10,0,10],
      ziters=[-10,0,10],
      returnAngles=False,
      returnPastedFilter=True
      ):
  '''
  This function is for the detection and marking of TT features in three dimensions. 

  Inputs:
    ttFilterName -> str. Name of the transverse tubule filter to be used
    ltFiltername -> str. Name of the longitudinal filter to be used
    lossFilterName -> str. Name of the tubule absence filter to be used
    ttPunishFilterName -> str. Name of the transverse tubule punishment filter to be used
    ltPunishFilterName -> str. Name of the longitudinal tubule punishment filter to be used NOTE: Delete?
    testImage -> str. Name of the image to be analyzed. NOTE: This image has previously been preprocessed by 
                   XXX routine.
    tag -> str. Base name of the written files 
    xiters -> list of ints. Rotations with which the filters will be rotated about the x axis (yz plane)
    yiters -> list of ints. Rotations with which the filters will be rotated about the y axis (xz plane)
    ziters -> list of ints. Rotations with which the filters will be rotated about the z axis (xy plane)
    returnAngles -> Bool. Whether or not to return the angles with which the hit experienced the greatest SNR
                      NOTE: Do I need to delete this? Not like I'll be able to get a colormap for this
    returnPastedFilter -> Bool. Whether or not to paste filter sized unit cells where detections occur.
                            This translates to much more intuitive hit markings.
  
  Outputs:
    TBD
  '''
  start = time.time()

  ### Read in preprocessed test image and store in inputs class for use in all subroutines
  inputs = empty()
  inputs.imgOrig = util.ReadImg(testImage)

  ### Form flattened iteration matrix containing all possible rotation combinations
  flattenedIters = []
  for i in xiters:
    for j in yiters:
      for k in ziters:
        flattenedIters.append( [i,j,k] )

  ### WT filtering
  if ttFilterName != None:
    WTresults = WT_Filtering(inputs,flattenedIters,ttFilterName,ttPunishFilterName,ttThresh,wtGamma,returnAngles)
    WTstackedHits = WTresults.stackedHits
  else:
    WTstackedHits = np.zeros_like(inputs.imgOrig)

  ### LT filtering
  if ltFilterName != None:
    LTresults = LT_Filtering(inputs,flattenedIters,ltFilterName,ltThresh,ltStdThresh,returnAngles)
    LTstackedHits = LTresults.stackedHits
  else:
    LTstackedHits = np.zeros_like(inputs.imgOrig)

  ### Loss filtering
  if taFilterName != None:
    ## form tubule absence flattened rotation matrix. Choosing to look at tubule absence at one rotation right now.
    taIters = [[0,0,0]]
    TAresults = Loss_Filtering(inputs,
                               taFilterName,
                               iters=taIters,
                               lossThresh=lossThresh,
                               lossStdThresh=lossStdThresh,
                               returnAngles=returnAngles)
    TAstackedHits = TAresults.stackedHits
  else:
    TAstackedHits = np.zeros_like(inputs.imgOrig)

  




  end = time.time()
  print "Time for algorithm to run:",start-end,"seconds"
  
  

def setupAnnotatedImage(annotatedName, baseImageName):
  '''
  Function to be used in conjunction with Myocyte().
  Uses the markPastedFilters() function to paste filters onto the annotated image.
  This is so we don't have to generate a new annotated image everytime we 
  change filter sizes.
  '''
  ### Read in images
  #baseImage = util.ReadImg(baseImageName,cvtColor=False)
  markedImage = util.ReadImg(annotatedName, cvtColor=False)
  
  ### Divide up channels of markedImage to represent hits
  wtHits, ltHits = markedImage[:,:,0],markedImage[:,:,1]
  wtHits[wtHits > 0] = 255
  ltHits[ltHits > 0] = 255
  # loss is already adequately marked so we don't want it ran through the routine
  lossHits = np.zeros_like(wtHits)
  coloredImage = markPastedFilters(lossHits,ltHits,wtHits,markedImage)
  # add back in the loss hits
  coloredImage[:,:,2] = markedImage[:,:,2]  

  ### Save image to run with optimizer routines
  newName = annotatedName[:-4]+"_pasted"+annotatedName[-4:]
  cv2.imwrite(newName,coloredImage)

  return newName
##
## Defines dataset for myocyte (MI) 
##
def Myocyte():
    # where to look for images
    root = "myoimages/"

    filter1TestName = root + "MI_annotation_testImg.png"
    filter1PositiveTest = root + "MI_annotation_trueImg.png"

    dataSet = optimizer.DataSet(
        root = root,
        filter1TestName = filter1TestName,
        filter1TestRegion = None,
        filter1PositiveTest = filter1PositiveTest,
        filter1PositiveChannel= 0,  # blue, WT 
        filter1Label = "TT",
        filter1Name = root+'WTFilter.png',          
        filter1Thresh=0.06, 
        
        filter2TestName = filter1TestName,
        filter2TestRegion = None,
        filter2PositiveTest = filter1PositiveTest,
        filter2PositiveChannel= 1,  # green, longi
        filter2Label = "LT",
        filter2Name = root+'newLTfilter.png',        
        filter2Thresh=0.38 
    )


    # flag to paste filters on the myocyte to smooth out results
    dataSet.pasteFilters = True

    return dataSet


def rocData(): 
  dataSet = Myocyte() 

  # rotation angles
  iters = [-25,-20,-15,-10,-5,0,5,10,15,20,25]

  root = "./myoimages/"

  # flag to turn on the pasting of unit cell on each hit
  dataSet.pasteFilters = True

  ## Testing TT first 
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'newSimpleWTFilter.png'
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.LoadFilter(root+"newSimpleWTPunishmentFilter.png")
  
  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        f1ts = np.linspace(0.1,0.45, 25),
        iters=iters,
        )

  ## Testing LT now
  dataSet.filter1PositiveChannel=1
  dataSet.filter1Label = "LT"
  dataSet.filter1Name = root+'LongitudinalFilter.png'
  #dataSet.filter1Name = root+'newLTfilter.png'
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='LT')  

  optimizer.GenFigROC_TruePos_FalsePos(
        dataSet,
        paramDict,
        filter1Label = dataSet.filter1Label,
        f1ts = np.linspace(0.01, 0.4, 25),
        iters=iters
      )

  ## Testing Loss
  dataSet.filter1PositiveChannel = 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+"LossFilter.png"
  optimizer.SetupTests(dataSet,meanFilter=True)
  paramDict = optimizer.ParamDict(typeDict='Loss')
  lossIters = [0,45]

  optimizer.GenFigROC_TruePos_FalsePos(
         dataSet,
         paramDict,
         filter1Label = dataSet.filter1Label,
         f1ts = np.linspace(0.005,0.1,25),
         iters=lossIters,
       )


###
### Function to calculate data for a full ROC for a given myocyte and return
### scores for each filter at given thresholds
###
def myocyteROC(data, myoName,
               threshes = np.linspace(5,30,10),
               iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25]
               ):
  root = "./myoimages/"

  ### WT
  # setup WT data in class structure
  data.filter1PositiveChannel= 0
  data.filter1Label = "TT"
  data.filter1Name = root + 'newSimpleWTFilter.png'
  optimizer.SetupTests(data,meanFilter=True)
  WTparams = optimizer.ParamDict(typeDict='WT')
  WTparams['covarianceMatrix'] = np.ones_like(data.filter1TestData)
  WTparams['mfPunishment'] = util.LoadFilter(root+"newSimpleWTPunishmentFilter.png")


  # write filter performance data for WT into hdf5 file
  optimizer.Assess_Single(data, 
                          WTparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_WT.h5",
                          display=False,
                          iters=iters)
  
  ### LT
  # setup LT data
  data.filter1PositiveChannel=1
  data.filter1Label = "LT"
  data.filter1Name = root+'LongitudinalFilter.png'
  optimizer.SetupTests(data,meanFilter=True)
  data.meanFilter = True
  LTparams = optimizer.ParamDict(typeDict='LT')

  # write filter performance data for LT into hdf5 file
  optimizer.Assess_Single(data, 
                          LTparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_LT.h5",
                          display=False,
                          iters=iters)

  ### Loss  
  # setup Loss data
  data.filter1PositiveChannel = 2
  data.filter1Label = "Loss"
  data.filter1Name = root+"LossFilter.png"
  optimizer.SetupTests(data,meanFilter=True)
  Lossparams = optimizer.ParamDict(typeDict='Loss')
  LossIters = [0,45]

  # write filter performance data for Loss into hdf5 file
  optimizer.Assess_Single(data, 
                          Lossparams, 
                          filter1Threshes=threshes, 
                          hdf5Name=myoName+"_Loss.h5",
                          display=False,
                          iters=LossIters)


###
### Function to convert from cv2's color channel convention to matplotlib's
###         
def switchBRChannels(img):
  newImg = img.copy()

  # ensuring to copy so that we don't accidentally alter the original image
  newImg[:,:,0] = img[:,:,2].copy()
  newImg[:,:,2] = img[:,:,0].copy()

  return newImg

def ReadResizeApplyMask(img,imgName,ImgTwoSarcSize,filterTwoSarcSize=25):
  # function to apply the image mask before outputting results
  maskName = imgName[:-4]; fileType = imgName[-4:]
  fileName = maskName+'_mask'+fileType
  mask = cv2.imread(fileName)                       
  try:
    maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  except:
    print "No mask named '"+fileName +"' was found. Circumventing masking."
    return img
  if ImgTwoSarcSize != None:
    scale = float(filterTwoSarcSize) / float(ImgTwoSarcSize)
    maskResized = cv2.resize(maskGray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
  else:
    maskResized = maskGray
  normed = maskResized.astype('float') / float(np.max(maskResized))
  normed[normed < 1.0] = 0
  dimensions = np.shape(img)
  if len(dimensions) < 3:
    combined = img * normed 
  else:
    combined = img
    for i in range(dimensions[2]):
      combined[:,:,i] = combined[:,:,i] * normed
  return combined

def assessContent(markedImg,imgName=None):
  # create copy
  imgCopy = markedImg.copy()
  # pull out channels
  wt = imgCopy[:,:,0]
  lt = imgCopy[:,:,1]
  loss = imgCopy[:,:,2]

  # get rid of everything that isn't a hit (hits are marked as 255)
  wt[wt != 255] = 0
  lt[lt != 255] = 0
  loss[loss != 255] = 0

  # normalize
  wtNormed = np.divide(wt, np.max(wt))
  ltNormed = np.divide(lt, np.max(lt))
  lossNormed = np.divide(loss, np.max(loss))

  # calculate content
  wtContent = np.sum(wtNormed)
  ltContent = np.sum(ltNormed)
  lossContent = np.sum(lossNormed)

  if isinstance(imgName, (str)):
    # if imgName is included, we normalize content to cell area
    dummy = np.multiply(np.ones_like(markedImg[:,:,0]), 255)
    mask = ReadResizeApplyMask(dummy,imgName,25,25)
    mask[mask <= 254] = 0
    mask[mask > 0] = 1
    cellArea = np.sum(mask,dtype=float)
    wtContent /= cellArea
    ltContent /= cellArea
    lossContent /= cellArea
    print "WT Content:", wtContent
    print "LT Content:", ltContent
    print "Loss Content:", lossContent
    print "Sum of Content:", wtContent+ltContent+lossContent
    # these should sum to 1 exactly but I'm leaving wiggle room
    assert (wtContent+ltContent+lossContent) < 1.2, ("Something went " 
            +"wrong with the normalization of content to the cell area calculated "
            +"by the mask. Double check the masking routine.") 
  else:
    print "WT Content:", wtContent
    print "LT Content:", ltContent
    print "Loss Content:", lossContent  

  return wtContent, ltContent, lossContent

def minDistanceROC(dataSet,paramDict,param1Range,param2Range,
                   param1="snrThresh",
                   param2="stdDevThresh",
                   FPthresh=0.1,
                   iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
                   ):
  '''
  Function that will calculate the minimum distance to the perfect detection point
  (0,1) on a ROC curve and return those parameters.
  '''
  perfectDetection = (0,1)

  distanceStorage = np.ones((len(param1Range),len(param2Range)),dtype=np.float32)
  TruePosStorage = np.ones_like(distanceStorage)
  FalsePosStorage = np.ones_like(distanceStorage)
  for i,p1 in enumerate(param1Range):
    paramDict[param1] = p1
    for j,p2 in enumerate(param2Range):
      paramDict[param2] = p2
      print "Param 1:",p1
      print "Param 2:",p2
      # having to manually assign the thresholds due to structure of TestParams function
      if param1 == "snrThresh":
        dataSet.filter1Thresh = p1
      elif param2 == "snrThresh":
        dataSet.filter1Thresh = p2
      posScore,negScore = optimizer.TestParams_Single(dataSet,paramDict,iters=iters)
      TruePosStorage[i,j] = posScore
      FalsePosStorage[i,j] = negScore
      if negScore < FPthresh:
        distanceFromPerfect = np.sqrt(((perfectDetection[0]-negScore)**2 +\
                                      (perfectDetection[1]-posScore)**2))
        distanceStorage[i,j] = distanceFromPerfect

  idx = np.unravel_index(distanceStorage.argmin(), distanceStorage.shape)
  optP1idx,optP2idx = idx[0],idx[1]
  optimumP1 = param1Range[optP1idx]
  optimumP2 = param2Range[optP2idx]
  optimumTP = TruePosStorage[optP1idx,optP2idx]
  optimumFP = FalsePosStorage[optP1idx,optP2idx]

  print ""
  print 100*"#"
  print "Minimum Distance to Perfect Detection:",distanceStorage.min()
  print "True Postive Rate:",optimumTP
  print "False Positive Rate:",optimumFP
  print "Optimum",param1,"->",optimumP1
  print "Optimum",param2,"->",optimumP2
  print 100*"#"
  print ""
  return optimumP1, optimumP2, distanceStorage

def optimizeWT():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'newSimpleWTFilter.png'
  optimizer.SetupTests(dataSet,meanFilter=True)

  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)
  paramDict['mfPunishment'] = util.LoadFilter(root+"newSimpleWTPunishmentFilter.png") 
  #snrThreshRange = np.linspace(0.01, 0.15, 35)
  #gammaRange = np.linspace(4., 25., 35)
  snrThreshRange = np.linspace(.1, 0.7, 20)
  gammaRange = np.linspace(1., 4., 20)

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,gammaRange,
                                                  param1="snrThresh",
                                                  param2="gamma", FPthresh=1.)

  plt.figure()
  plt.imshow(distToPerfect)
  plt.colorbar()
  plt.gcf().savefig("ROC_Optimization_WT.png")
  

def optimizeLT():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 1
  dataSet.filter1Label = "LT"
  dataSet.filter1Name = root+'LongitudinalFilter.png'
  optimizer.SetupTests(dataSet)

  paramDict = optimizer.ParamDict(typeDict='LT')
  snrThreshRange = np.linspace(0.4, 0.8, 20)
  stdDevThreshRange = np.linspace(0.05, 0.4, 20)

  FPthresh = 1.

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,stdDevThreshRange,
                                                  param1="snrThresh",
                                                  param2="stdDevThresh",
                                                  FPthresh=FPthresh)

  plt.figure()
  plt.imshow(distToPerfect)
  plt.colorbar()
  plt.gcf().savefig("ROC_Optimization_LT.png")

def optimizeLoss():
  root = "./myoimages/"
  dataSet = Myocyte()
  dataSet.filter1PositiveChannel= 2
  dataSet.filter1Label = "Loss"
  dataSet.filter1Name = root+'LossFilter.png'
  optimizer.SetupTests(dataSet)

  paramDict = optimizer.ParamDict(typeDict='Loss')
  snrThreshRange = np.linspace(0.05,0.3, 20)
  stdDevThreshRange = np.linspace(0.05, 0.2, 20)

  optimumSNRthresh, optimumGamma, distToPerfect= minDistanceROC(dataSet,paramDict,
                                                  snrThreshRange,stdDevThreshRange,
                                                  param1="snrThresh",
                                                  param2="stdDevThresh",
                                                  FPthresh=1.)

  plt.figure()
  plt.imshow(distToPerfect)
  plt.colorbar()
  plt.gcf().savefig("ROC_Optimization_Loss.png")

# function to validate that code has not changed since last commit
def validate(testImage="./myoimages/MI_D_78_processed.png",
             display=False
             ):
  # run algorithm
  markedImg = giveMarkedMyocyte(testImage=testImage)

  if display:
    plt.figure()
    plt.imshow(markedImg)
    plt.show()

  # calculate wt, lt, and loss content  
  wtContent, ltContent, lossContent = assessContent(markedImg)

  assert(abs(wtContent - 52594) < 1), "WT validation failed."
  assert(abs(ltContent - 11687) < 1), "LT validation failed."
  assert(abs(lossContent - 12752) < 1), "Loss validation failed."
  print "PASSED!"

# A minor validation function to serve as small tests between commits
def minorValidate(testImage="./myoimages/MI_D_73_annotation.png",
                  ImgTwoSarcSize=25, #img is already resized to 25 px
                  iters=[-10,0,10],
                  display=False):

  # run algorithm
  markedImg = giveMarkedMyocyte(testImage=testImage, 
                                ImgTwoSarcSize=ImgTwoSarcSize,iters=iters)
  if display:
    plt.figure()
    plt.imshow(markedImg)
    plt.show()

  # assess content
  wtContent, ltContent, lossContent = assessContent(markedImg) 
  
  print "WT Content:",wtContent
  print "Longitudinal Content", ltContent
  print "Loss Content", lossContent

  val = 18722 
  assert(abs(wtContent - val) < 1),"%f != %f"%(wtContent, val)       
  val = 3669
  assert(abs(ltContent - val) < 1),"%f != %f"%(ltContent, val) 
  val = 1420
  assert(abs(lossContent - val) < 1),"%f != %f"%(lossContent, val)
  print "PASSED!"


###
### Function to test that the optimizer routines that assess positive and negative
### filter scores are working correctly.
###
def scoreTest():
  dataSet = Myocyte() 

  ## Testing TT first 
  dataSet.filter1PositiveChannel=0
  dataSet.filter1Label = "TT"
  dataSet.filter1Name = root+'WTFilter.png'
  optimizer.SetupTests(dataSet)
  dataSet.filter1Thresh = 5.5

  paramDict = optimizer.ParamDict(typeDict='WT')
  paramDict['covarianceMatrix'] = np.ones_like(dataSet.filter1TestData)

  filter1PS,filter1NS = optimizer.TestParams_Single(
    dataSet,
    paramDict,
    iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
    display=False)  
    #display=True)  

  print filter1PS, filter1NS

  val = 0.926816518557
  assert((filter1PS - val) < 1e-3), "Filter 1 Positive Score failed"
  val = 0.342082872458
  assert((filter1NS - val) < 1e-3), "Filter 1 Negative Score failed"
  print "PASSED"


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
tag = "default_" 
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):

    ### Validation Routines
    if(arg=="-validate"):
      print "Consider developing a more robust behavior test"
      validate()
      quit()

    if(arg=="-minorValidate"):
      minorValidate()
      quit()

    if(arg=="-scoretest"):
      scoreTest()             
      quit()
    

    ### Figure Generation Routines

    # this function will generate input data for the current fig #3 in the paper 
    if(arg=="-WT"):               
      WT_results()
      quit()

    if(arg=="-HF"):               
      HF_results()
      quit()

    if(arg=="-MI"):               
      MI_results()
      quit()

    if(arg=="-tissueComparison"):               
      tissueComparison(fullAnalysis=True)
      quit()

    if(arg=="-figAngle"):
      figAngleHeterogeneity()
      quit()

    if(arg=="-full_ROC"):
      figS1()
      quit()

    if(arg=="-tissueBloodVessel"):
      tissueBloodVessel()
      quit()

    if(arg=="-algComparison"):
      algComparison()
      quit()

    if(arg=="-yaml"):
      YAML_example()
      quit()

    # generates all figs
    if(arg=="-allFigs"):
      WT_results()     
      HF_results()     
      MI_results()
      tissueComparison()     
      full_ROC()
      tissueBloodVessel()
      algComparison()
      YAML_example()
      quit()

    if(arg=="-workflowFig"):
      saveWorkflowFig()
      quit()

    ### Testing/Optimization Routines
    if(arg=="-roc"): 
      rocData()
      quit()

    if(arg=="-optimizeWT"):
      optimizeWT()
      quit()

    if(arg=="-optimizeLT"):
      optimizeLT()
      quit()

    if(arg=="-optimizeLoss"):
      optimizeLoss()
      quit()
	   
    if(arg=="-test"):
      giveMarkedMyocyte(      
        ttFilterName=sys.argv[i+1],
        ltFilterName=sys.argv[i+2],
        testImage=sys.argv[i+3],           
        ttThresh=np.float(sys.argv[i+4]),           
        ltThresh=np.float(sys.argv[i+5]),
        gamma=np.float(sys.argv[i+6]),
        ImgTwoSarcSize=(sys.argv[i+7]),
	tag = tag,
	writeImage = True)            
      quit()

    if(arg=="-testMyocyte"):
      testImage = sys.argv[i+1]
      giveMarkedMyocyte(testImage=testImage,
                        tag="Testing",
                        writeImage=True)
      quit()

    if(arg=="-analyzeAllMyo"):
      analyzeAllMyo()
      quit()

    if(arg=="-analyzeSingleMyo"):
      name = sys.argv[i+1]
      twoSarcSize = float(sys.argv[i+2])
      analyzeSingleMyo(name,twoSarcSize)
      quit()

    if(arg=="-testTissue"):
      name = "testingNotchFilter.png"
      giveMarkedMyocyte(testImage=name,
                        tag="TestingNotchedFilter",
                        iters=[-5,0,5],
                        returnAngles=False,
                        writeImage=True,
                        useGPU=True)
      quit()

    if(arg=="-analyzeDirectory"):
      root = sys.argv[i+1]
      analyzeAllMyo(root=root)
      quit()

    ### Additional Arguments
    if(arg=="-tag"):
      tag = sys.argv[i+1]

    if(arg=="-noPrint"):
      sys.stdout = open(os.devnull, 'w')

    if(arg=="-shaveFig"):
      fileName = sys.argv[i+1]
      try:
        padY = int(sys.argv[i+2])
        padX = int(sys.argv[i+3])
        whiteSpace = int(sys.argv[i+4])
      except:
        padY = None
        padX = None
        whiteSpace = None
      shaveFig(fileName,padY=padY,padX=padX,whiteSpace=whiteSpace)
      quit()

  raise RuntimeError("Arguments not understood")
