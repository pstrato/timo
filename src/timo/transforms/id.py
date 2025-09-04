from timo.transform import Transform


class Id(Transform):

    def name(self, input_shape, output_shape):
        return "Id"

    def output_shape(self, input_shape):
        return input_shape
