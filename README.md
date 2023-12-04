# ml2grpj
OCR Application for automated information input, for this project, tested on StudentID cards

Includes 3 major parts:

  ID Card Extraction from raw image
  
  Words and Letters Extraction
  
  Classification
  

ID Card Extraction:

  Extract image corners (round corners, not common sharp corners)
  
  Perspective Warp


Words and Letters Extraction

  Words Extraction:
  
    Extract lines of information from pre-measured coordinates
    
    Extract words from lines of information
    
  Letters Extraction:
  
    Extract letters (with small symbols) from extracted words


Classification: Very simple deep learning networks, as pre-processing was done well

  Letters: 95%
  
  Numbers: 95%
  

Conclusion: Pretty naive project, still works pretty good with the right set up and equipment


