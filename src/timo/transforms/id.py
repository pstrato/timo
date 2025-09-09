from timo.transform import Transform


class Id(Transform):
    def validate(self, inputs):
        assert len(inputs) == 1

    def name(self, inputs, output_shapes):
        return "Id"

    def output_shapes(self, inputs):
        return inputs[0].shapes
