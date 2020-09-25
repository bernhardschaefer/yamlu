from yamlu.img import AnnotatedImage, Annotation, BoundingBox


def test_annotation_repr():
    a = Annotation("test", BoundingBox(0, 5, 10, 10), text="foo")
    print(a)
    assert str(a) == "Annotation(category='test', bb=BoundingBox(t=0.00,l=5.00,b=10.00,r=10.00), text=foo)"


def test_annotated_image_repr():
    a1 = Annotation("cat1", BoundingBox(0, 5, 10, 10), text="foo")
    a2 = Annotation("cat2", BoundingBox(0, 0, 10, 10), text="bla", arg2="test")
    a = AnnotatedImage("test.png", width=100, height=200, annotations=[a1, a2])

    print(a)
    print(a.annotations)
