from PIL import Image, ImageDraw
import numpy as np
import random
import os.path
import pickle
from IPython.display import display

def ComputeSSD(TODOPatch, TODOMask, textureIm, patchL):
    patch_rows, patch_cols, patch_bands = np.shape(TODOPatch)
    tex_rows, tex_cols, tex_bands = np.shape(textureIm)
    ssd_rows = tex_rows - 2 * patchL
    ssd_cols = tex_cols - 2 * patchL
    SSD = np.zeros((ssd_rows,ssd_cols))
    
    # Turn the uint8 images into floats to perform precise calculations
    # (synth patch is the patch we will synthesize AKA TODOpatch)
    synth_patch = np.copy(TODOPatch).astype('float64')
    sample_img = np.copy(textureIm).astype('float64')

    # Turn the image into greyscale
    synth_patch = np.mean(synth_patch, axis=2)
    sample_img = np.mean(sample_img, axis=2)

    # NOTE: I commented out the gaussian part because Q2 says we are ignoring
    # the Gaussian weighted window....
    # Create a gaussian distribution the same size as the patch for spatial
    # weighting later. (I'll use it for element-wise operation when computing SSD)
    sigma = patchL / 3
    gauss_1d = np.arange(-patchL, patchL + 1, dtype='float64')
    gauss_1d = np.vectorize(lambda x: np.exp(- (x ** 2) / (2 * sigma ** 2)))(gauss_1d)
    gauss_1d = gauss_1d[:, None] # I want this as a column vector
    gauss_2d = gauss_1d @ gauss_1d.T
    
    # Normalize so the sum is 1.
    gauss_normalized_2d = gauss_2d / np.sum(gauss_2d)

    for r in range(ssd_rows):
        for c in range(ssd_cols):
            # Compute sum square difference between textureIm and TODOPatch
            # for all pixels where TODOMask = 0, and store the result in SSD
            
            # Get column index of the patch borders in the sample image.
            left_col_sample_patch = c
            right_col_sample_patch = c + 2 * patchL

            # Get row index of the patch borders in the sample image.
            top_row_sample_patch = r
            bottom_row_sample_patch = r + 2 * patchL
            
            # Get all pixels within the patch centered at the SSD location (r, c)
            sample_patch = sample_img[top_row_sample_patch : bottom_row_sample_patch + 1, 
                                     left_col_sample_patch : right_col_sample_patch + 1]
            
            # Remove unknown pixels from sample patch 
            # (using inverted TODOmask because 1 represents missing pixels)
            sample_patch = sample_patch * (1.0 - TODOMask)
            
            # NOTE:  that this technique of copying a whole patch is much faster than 
            # copying just the center pixel as suggested in the original Efros and 
            # Leung paper. However, the results are not quite as good. We are also 
            # ignoring the use of a Gaussian weighted window as described in their paper.
            # SSD[r, c] = np.sum(((sample_patch - synth_patch) ** 2) * gauss_normalized_2d)

            # Compute SSD.
            SSD[r, c] = np.sum((sample_patch - synth_patch) ** 2)
        
    return SSD

def CopyPatch(imHole,TODOMask,textureIm,iPatchCenter,jPatchCenter,iMatchCenter,jMatchCenter,patchL):
    # it looks like i is representing the row, and j represents the 
    # column in this instance...
    # Get row index of the patch borders for the synthetic patch
    top_row_synth_patch = iPatchCenter - patchL
    bottom_row_synth_patch = iPatchCenter + patchL

    # Get col index of the patch borders for the synthetic patch
    left_col_synth_patch = jPatchCenter - patchL
    right_col_synth_patch = jPatchCenter + patchL

    # get the pixels inside the synthetic patch
    synth_patch = imHole[top_row_synth_patch : bottom_row_synth_patch + 1,
                            left_col_synth_patch : right_col_synth_patch + 1]
    
    # Get row index of the patch borders for the selected patch
    top_row_selected_patch = iMatchCenter - patchL
    bottom_row_selected_patch = iMatchCenter + patchL

    # Get col index of the patch borders for the selected patch
    left_col_selected_patch = jMatchCenter - patchL
    right_col_selected_patch = jMatchCenter + patchL

    # get the pixels inside the selected patch
    selected_patch = textureIm[top_row_selected_patch : bottom_row_selected_patch + 1,
                               left_col_selected_patch : right_col_selected_patch + 1]
    
    # Merge the known pixels from synth_patch with the unkown pixels 
    # taken from selected_patch
    synth_patch = synth_patch + selected_patch * TODOMask[:,:,None]

    # Copy the selected patch selectPatch into the image containing
    # the hole imHole for each pixel where TODOMask = 1.
    # fill in imHole with the merged patch
    imHole[top_row_synth_patch : bottom_row_synth_patch + 1,
            left_col_synth_patch : right_col_synth_patch + 1] = synth_patch

    return imHole

def DrawBox(im,x1,y1,x2,y2):
    draw = ImageDraw.Draw(im)
    draw.line((x1,y1,x1,y2),fill="white",width=1)
    draw.line((x1,y1,x2,y1),fill="white",width=1)
    draw.line((x2,y2,x1,y2),fill="white",width=1)
    draw.line((x2,y2,x2,y1),fill="white",width=1)
    del draw
    return im

def Find_Edge(hole_mask):
    [cols, rows] = np.shape(hole_mask)
    edge_mask = np.zeros(np.shape(hole_mask))
    for y in range(rows):
        for x in range(cols):
            if (hole_mask[x,y] == 1):
                if (hole_mask[x-1,y] == 0 or
                        hole_mask[x+1,y] == 0 or
                        hole_mask[x,y-1] == 0 or
                        hole_mask[x,y+1] == 0):
                    edge_mask[x,y] = 1
    return edge_mask


##############################################################################
#                           Main script starts here                          #
##############################################################################

#
# Constants
#

# Change patchL to change the patch size used (patch size is 2 *patchL + 1)
patchL = 10
patchSize = 2*patchL+1

# Standard deviation for random patch selection
randomPatchSD = 1

# Display results interactively
#
# Read input image
#

im = Image.open('./donkey.jpg').convert('RGB')
im_array = np.asarray(im, dtype=np.uint8)
imRows, imCols, imBands = np.shape(im_array)

#
# Define hole and texture regions.  This will use files fill_region.pkl and
#   texture_region.pkl, if both exist, otherwise user has to select the regions.
if os.path.isfile('fill_region.pkl') and os.path.isfile('texture_region.pkl'):
    fill_region_file = open('fill_region.pkl', 'rb')
    fillRegion = pickle.load( fill_region_file )
    fill_region_file.close()

    texture_region_file = open('texture_region.pkl', 'rb')
    textureRegion = pickle.load( texture_region_file )
    texture_region_file.close()
else:
    # ask the user to define the regions
    print("Specify the fill and texture regions using polyselect.py")
    exit()

#
# Get coordinates for hole and texture regions
#

fill_indices = fillRegion.nonzero()
nFill = len(fill_indices[0])                # number of pixels to be filled
iFillMax = max(fill_indices[0])
iFillMin = min(fill_indices[0])
jFillMax = max(fill_indices[1])
jFillMin = min(fill_indices[1])
assert((iFillMin >= patchL) and
        (iFillMax < imRows - patchL) and
        (jFillMin >= patchL) and
        (jFillMax < imCols - patchL)) , "Hole is too close to edge of image for this patch size"

texture_indices = textureRegion.nonzero()
iTextureMax = max(texture_indices[0])
iTextureMin = min(texture_indices[0])
jTextureMax = max(texture_indices[1])
jTextureMin = min(texture_indices[1])
textureIm   = im_array[iTextureMin:iTextureMax+1, jTextureMin:jTextureMax+1, :]
texImRows, texImCols, texImBands = np.shape(textureIm)
assert((texImRows > patchSize) and
        (texImCols > patchSize)) , "Texture image is smaller than patch size"

#
# Initialize imHole for texture synthesis (i.e., set fill pixels to 0)
#

imHole = im_array.copy()
imHole[fill_indices] = 0
showResults = True
#
# Is the user happy with fillRegion and textureIm?
#
if showResults == True:
    # original
    #im.show()
    display(im)
    # convert to a PIL image, show fillRegion and draw a box around textureIm
    im1 = Image.fromarray(imHole).convert('RGB')
    im1 = DrawBox(im1,jTextureMin,iTextureMin,jTextureMax,iTextureMax)
    display(im1)
    #im1.show()
    print("Are you happy with this choice of fillRegion and textureIm?")
    Yes_or_No = False
    while not Yes_or_No:
        answer = input("Yes or No: ")
        if answer == "Yes" or answer == "No":
            Yes_or_No = True
    assert answer == "Yes", "You must be happy. Please try again."

#
# Perform the hole filling
#

while (nFill > 0):
    print("Number of pixels remaining = " , nFill)

    # Set TODORegion to pixels on the boundary of the current fillRegion
    TODORegion = Find_Edge(fillRegion)
    edge_pixels = TODORegion.nonzero()
    nTODO = len(edge_pixels[0])

    while(nTODO > 0):

        # Pick a random pixel from the TODORegion
        index = np.random.randint(0,nTODO)
        iPatchCenter = edge_pixels[0][index]
        jPatchCenter = edge_pixels[1][index]

        # Define the coordinates for the TODOPatch
        TODOPatch = imHole[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1,:]
        TODOMask = fillRegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1]

        #
        # Compute masked SSD of TODOPatch and textureIm
        #
        ssdIm = ComputeSSD(TODOPatch, TODOMask, textureIm, patchL)

        # Randomized selection of one of the best texture patches
        ssdIm1 = np.sort(np.copy(ssdIm),axis=None)
        ssdValue = ssdIm1[min(round(abs(random.gauss(0,randomPatchSD))),np.size(ssdIm1)-1)]
        ssdIndex = np.nonzero(ssdIm==ssdValue)
        iSelectCenter = ssdIndex[0][0]
        jSelectCenter = ssdIndex[1][0]

        # adjust i, j coordinates relative to textureIm
        iSelectCenter = iSelectCenter + patchL
        jSelectCenter = jSelectCenter + patchL
        selectPatch = textureIm[iSelectCenter-patchL:iSelectCenter+patchL+1,jSelectCenter-patchL:jSelectCenter+patchL+1,:]

        #
        # Copy patch into hole
        #
        imHole = CopyPatch(imHole,TODOMask,textureIm,iPatchCenter,jPatchCenter,iSelectCenter,jSelectCenter,patchL)

        # Update TODORegion and fillRegion by removing locations that overlapped the patch
        TODORegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1] = 0
        fillRegion[iPatchCenter-patchL:iPatchCenter+patchL+1,jPatchCenter-patchL:jPatchCenter+patchL+1] = 0

        edge_pixels = TODORegion.nonzero()
        nTODO = len(edge_pixels[0])

    fill_indices = fillRegion.nonzero()
    nFill = len(fill_indices[0])

#
# Output results
#
if showResults == True:
    display(Image.fromarray(imHole).convert('RGB'))
Image.fromarray(imHole).convert('RGB').save('results.jpg')
