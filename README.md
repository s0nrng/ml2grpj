# ml2grpj
OCR Application for automated information input, for this project, tested on StudentID cards

Includes 3 major parts:

_ID Card Extraction from raw image
  
_Words and Letters Extraction
  
_Classification
  

ID Card Extraction:

_Extract image corners (round corners, not common sharp corners)
  
_Perspective Warp


Words and Letters Extraction

_Words Extraction:
  
__Extract lines of information from pre-measured coordinates
    
__Extract words from lines of information
    
_Letters Extraction:
  
__Extract letters (with small symbols) from extracted words


Classification: Very simple deep learning networks, as pre-processing was done well

_Letters: 95%
  
_Numbers: 95%
  

Conclusion: Pretty naive project, still works pretty good with the right set up and equipment


