import cv2

def yolo_to_absolute(image, line):
    """Convert YOLO format to absolute pixel values."""
    height, width = image.shape[:2]
    
    # Parsing YOLO format data
    class_id, x_center, y_center, box_width, box_height = map(float, line.split())
    class_id = int(class_id)
    
    # Convert YOLO center coordinates (relative values between 0 and 1) to absolute pixel values
    x_center_abs = int(x_center * width)
    y_center_abs = int(y_center * height)

    # Convert YOLO box dimensions (relative values between 0 and 1) to absolute pixel values
    box_width_abs = int(box_width * width)
    box_height_abs = int(box_height * height)

    # Calculate top-left and bottom-right coordinates for the bounding box
    x1 = int(x_center_abs - (box_width_abs / 2))
    y1 = int(y_center_abs - (box_height_abs / 2))
    x2 = int(x_center_abs + (box_width_abs / 2))
    y2 = int(y_center_abs + (box_height_abs / 2))
    
    return class_id, (x1, y1, x2, y2)

def draw_boxes_on_image(image_path, annotations_path):
    # Read the image
    image = cv2.imread(image_path)

    # Read annotations from the txt file
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()
    
    print(annotations)
    
    # Loop over each line (annotation) in the YOLO formatted annotations
    for line in annotations:
        class_id, (x1, y1, x2, y2) = yolo_to_absolute(image, line.strip())
        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color, thickness=2
        # You can also put class_id or some text on the image if needed using cv2.putText

    # Display the image
    # cv2.imshow("Image with Bounding Boxes", image)
    cv2.imwrite("bounding_box_image.jpg", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Sample usage
image_path = "images/000619_png_jpg.rf.75d5b762b01e8ea4c8dceb068e378522.jpg"
annotations_path = "labels/000619_png_jpg.rf.75d5b762b01e8ea4c8dceb068e378522.txt"
draw_boxes_on_image(image_path, annotations_path)
