from manim import *
import itertools as it


# background color
config.background_color = "#1E1E1E"
config.tex_template.add_to_preamble(r"\usepackage{physics}") # to enables physics extra latex package, can use \pdv for partial derivative 

def ARROWS(layers):
    layer_neurons = {}
    for i, _ in enumerate(layers):
        layer_neurons[i] = VGroup()

    for i, layer in enumerate(layers):
        for x in range(0,layer):
            for y in range(0,layers[i+1]):
                arrow = Arrow(start=x, end=y, max_tip_length_to_length_ratio=0.03, stroke_width=3)
                layer_neurons[i].add(arrow)
    return layer_neurons.values



class Intro(Scene): # SCRAPPED 
    def construct(self):

        X = MathTex("X", font_size=300).shift(LEFT*4)
        Y = MathTex("Y", font_size=300).shift(RIGHT*4)
        arrow = Arrow(start=X.get_center() + [1,0,0], end=Y.get_center() + [-1,0,0])

        self.add(X,Y,arrow)

class Network(Scene):
    def construct(self):
        
        input_C0 = Circle(radius=0.2, color=BLUE_C, stroke_color=BLACK, fill_opacity=1)
        input_C1 = Circle(radius=0.2, color=BLUE_C, stroke_color=BLACK, fill_opacity=1)
        input_C2 = Circle(radius=0.2, color=BLUE_C, stroke_color=BLACK, fill_opacity=1)
        
        hidden_C0 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)

        output_C0 = Circle(radius=0.2, color=RED_C, stroke_color=BLACK, fill_opacity=1)
        output_C1 = Circle(radius=0.2, color=RED_C, stroke_color=BLACK, fill_opacity=1)
        output_C2 = Circle(radius=0.2, color=RED_C, stroke_color=BLACK, fill_opacity=1)
        output_C3 = Circle(radius=0.2, color=RED_C, stroke_color=BLACK, fill_opacity=1)

        nn = VGroup(input_C0, hidden_C0, output_C0).arrange(buff=5)

        inputs = VGroup(input_C1, input_C2).arrange(DOWN, buff=1).move_to(input_C0)
        outputs = VGroup(output_C1, output_C2, output_C3).arrange(DOWN, buff=1).move_to(output_C0)

        arrow1 = Arrow(start=input_C0, end=hidden_C0, max_tip_length_to_length_ratio=0.05)
        arrow2 = Arrow(start=hidden_C0, end=output_C0, max_tip_length_to_length_ratio=0.05)
        arrows = VGroup(arrow1, arrow2)

        input = Tex("Input layer","Inputs layer", color=BLUE_C)
        hidden = Tex("Hidden layer", "Hidden layers" )
        output = Tex("Output layer", "Outputs layer", color=RED_C)



        text = VGroup(input[0], hidden[0], output[0]).arrange(buff=2).to_edge(DOWN)

        X = MathTex("X", font_size=300, color=BLUE_C).shift(LEFT*4)
        Y = MathTex("Y", font_size=300, color=RED_C).shift(RIGHT*4)
        arrow_main = Arrow(start=X.get_center() + [1,0,0], end=Y.get_center() + [-1,0,0])


        hidden_l1__j1 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)
        hidden_l1__j2 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)
        hidden_l1__j3 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)
        hidden_l1__j4 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)
        hidden_l1 = VGroup(hidden_l1__j1, hidden_l1__j2, hidden_l1__j3, hidden_l1__j4).arrange(DOWN, buff=1).move_to(LEFT*2)

        hidden_l2__j1 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)
        hidden_l2__j2 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)
        hidden_l2 = VGroup(hidden_l2__j1, hidden_l2__j2).arrange(DOWN, buff=1).move_to(RIGHT*2)


        inputs_l1 = VGroup()
        for i in hidden_l1:
            arrow = Arrow(start=input_C0, end=i, max_tip_length_to_length_ratio=0.05)
            inputs_l1.add(arrow)

        l1_l2 = VGroup()
        for i in hidden_l2:
            for j in hidden_l1:
                arrow = Arrow(start=j, end=i, max_tip_length_to_length_ratio=0.05)
                l1_l2.add(arrow)
        
        l2_outputs = VGroup()
        for i in hidden_l2:
            arrow = Arrow(start=i, end=output_C0, max_tip_length_to_length_ratio=0.05)
            l2_outputs.add(arrow)  



        inputs_l1_v2 = VGroup()
        for i in hidden_l1:
            for j in inputs:
                arrow = Arrow(start=j, end=i, max_tip_length_to_length_ratio=0.05)
                inputs_l1_v2.add(arrow)

        l2_outputs_v2 = VGroup()
        for i in hidden_l2:
            for j in outputs:
                arrow = Arrow(start=i, end=j, max_tip_length_to_length_ratio=0.05)
                l2_outputs_v2.add(arrow)  
        

        self.play(AnimationGroup(Write(X), Write(Y), Write(arrow_main)))
        self.wait()
        self.play(AnimationGroup(Transform(X, nn[0]), Transform(Y, nn[2]), Transform(arrow_main, arrows), Write(nn[1],run_time = 2)), run_time=2)
        self.play(Write(text))
        self.wait()
        self.play(FadeOut(arrow_main,arrow2,hidden_C0))
        self.play(AnimationGroup(Write(hidden_l1), Write(hidden_l2), Write(inputs_l1), Write(l1_l2), Write(l2_outputs), TransformMatchingShapes(hidden[0], hidden[1], run_time=1)))
        self.wait()
        self.play(Transform(X, inputs),Transform(Y, outputs), Transform(inputs_l1, inputs_l1_v2),
                 Transform(l2_outputs, l2_outputs_v2),TransformMatchingShapes(input[0],input[1]), TransformMatchingShapes(output[0],output[1]))
        self.wait()


class Forward(Scene):
    def construct(self):
            
            input_C0 = Circle(radius=0.1, color=BLUE_C, stroke_color=BLACK, fill_opacity=1, stroke_width=2)
            hidden_C0 = Circle(radius=0.1, color=WHITE, stroke_color=BLACK, fill_opacity=1, stroke_width=2)
            output_C0 = Circle(radius=0.1, color=RED_C, stroke_color=BLACK, fill_opacity=1, stroke_width=2)
            network0 = VGroup(input_C0, hidden_C0, output_C0).arrange(buff=5).move_to(UP * 2.5)

            arrow_C0_1 = Arrow(start=input_C0, end=hidden_C0, max_tip_length_to_length_ratio=0.03, stroke_width=3)
            arrow_C0_2 = Arrow(start=hidden_C0, end=output_C0, max_tip_length_to_length_ratio=0.03, stroke_width=3)
            arrow0 = VGroup(arrow_C0_1, arrow_C0_2)

            self.play(AnimationGroup(Write(network0),Create(arrow0) ))
            self.wait()

            equation0 = MathTex(r"z &= wx + b  \\ a &= g(z)", font_size=50).to_edge(LEFT)
            equation0[0][3].set_fill(color=BLUE_C)
            equation0[0][6].set_fill(color=RED_C)

            self.play(Write(equation0))
            self.wait()

            hidden_C1 = Circle(radius=0.1, color=WHITE, stroke_color=BLACK, fill_opacity=1, stroke_width=2)
            hidden_C2 = Circle(radius=0.1, color=WHITE, stroke_color=BLACK, fill_opacity=1, stroke_width=2)

            network1 = VGroup(hidden_C0, hidden_C1, hidden_C2).arrange(buff=5).move_to(UP * 2.5)
            arrow1 = ARROWS([1,3,1])

            self.play(AnimationGroup(FadeOut(arrow0), Write(arrow1)))


class test(Scene):
    def construct(self):
        mynetwork = NeuralNetworkMobject([1,3,1])
        mynetwork.label_inputs('x')
        mynetwork.label_outputs('y')
        mynetwork.label_outputs_text(['Number'])
        MathTex
        self.play(Write(mynetwork))
        self.wait

# A customizable Sequential Neural Network, copied from https://www.youtube.com/watch?v=HnIeAP--vWc
class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,

        "output_neuron_color": RED_C,
        "input_neuron_color": BLUE_C,
        "hidden_layer_neuron_color": WHITE,

        "neuron_stroke_width": 2,
        "neuron_fill_color": GREEN,
        "edge_color": LIGHT_GREY,
        "edge_stroke_width": 2,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
        "max_shown_neurons": 16,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 1
    }
    # Constructor with parameters of the neurons in a list
    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)
    # Helper method for constructor
    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange_submobjects(RIGHT, buff=self.CONFIG['layer_to_layer_buff'])
        self.layers = layers
        if self.CONFIG['include_output_labels']:
            self.label_outputs_text()
    # Helper method for constructor
    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.CONFIG['output_neuron_color']
        if index == 0:
            return self.CONFIG['input_neuron_color']
        else:
            return self.CONFIG['hidden_layer_neuron_color']
    # Helper method for constructor
    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.CONFIG['max_shown_neurons']:
            n_neurons = self.CONFIG['max_shown_neurons']
        neurons = VGroup(*[
            Circle(
                radius=self.CONFIG['neuron_radius'],
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.CONFIG['neuron_stroke_width'],
                fill_color=BLACK,
                fill_opacity=self.CONFIG['neuron_fill_opacity'],
            )
            for x in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=self.CONFIG['neuron_to_neuron_buff']
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = MathTex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.CONFIG['brace_for_large_layers']:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer
    # Helper method for constructor
    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)
    # Helper method for constructor
    def get_edge(self, neuron1, neuron2):
        if self.CONFIG['arrow']:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.CONFIG['neuron_radius'],
                stroke_color=self.CONFIG['edge_color'],
                stroke_width=self.CONFIG['edge_stroke_width'],
                tip_length=self.CONFIG['arrow_tip_size']
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.CONFIG['neuron_radius'],
            stroke_color=self.CONFIG['edge_color'],
            stroke_width=self.CONFIG['edge_stroke_width'],
        )
    
    # Labels each input neuron with a char l or a LaTeX character
    def label_inputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = MathTex(f"{l}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each output neuron with a char l or a LaTeX character
    def label_outputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = MathTex(f"{l}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each neuron in the output layer with text according to an output list
    def label_outputs_text(self, outputs):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = MathTex(outputs[n])
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + label.get_width()/2)*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels the hidden layers with a char l or a LaTeX character
    def label_hidden_layers(self, l):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = MathTex(f"{l}_{n + 1}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)