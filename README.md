In this repository there are scripts to perform muscle ultrasound quantitative analysis:
a) echointensity.py - calculates echo intensity (the brightness) of the muscle 
b) glcm.py - calculates the GLCM (gray-level co-occurrence matrix)
c) rlm_normalized_square.py - calculates RLM (run-length matrix)
d) lbp.py - calculates LBP (local binary pattern)

Organize your muscles in the "images" folder and masks in the "masks" folder. Masks should be in 0-255 format (black-background, white-mask). By default pixel distance is 1 and angles are 0, 45, 90 and 135 but you can change it.

Good luck:)
