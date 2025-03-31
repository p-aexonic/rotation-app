from rotation_handler import RotationHandler

rh = RotationHandler()

# Image Test
rh.handle_image_rotation('test_files/Landscape_2.jpg', 'test_files/Landscape_2_fixed.jpg')

# PDF Test
rh.handle_pdf_rotation('test_files/Get_Started_With_Smallpdf-rotated.pdf', 'test_files/Get_Started_With_Smallpdf_fixed.pdf')
