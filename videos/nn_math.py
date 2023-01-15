from manim import *

# background color
config.background_color = "#1E1E1E"
config.tex_template.add_to_preamble(r"\usepackage{physics}") # to enables physics extra latex package, can use \pdv for partial derivative 


class Intro(Scene):
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

        input = Tex(r"Input layer", color=BLUE_C)
        hidden = Tex(r"Hidden layer")
        hiddens = Tex(r"Hidden layers").to_edge(DOWN)
        output = Tex(r"Output layer", color=RED_C)
        text = VGroup(input, hidden, output).arrange(buff=2).to_edge(DOWN)

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
        self.play(AnimationGroup(Write(hidden_l1), Write(hidden_l2), Write(inputs_l1), Write(l1_l2), Write(l2_outputs), TransformMatchingShapes(hidden, hiddens), run_time=5))
        self.wait()
        self.play(Transform(X, inputs),Transform(Y, outputs), Transform(inputs_l1, inputs_l1_v2), Transform(l2_outputs, l2_outputs_v2))
        self.wait()





