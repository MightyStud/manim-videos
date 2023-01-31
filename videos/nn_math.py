from manim import *
import itertools as it
import numpy as np


# background color
config.background_color = "#1E1E1E"
config.tex_template.add_to_preamble(r"\usepackage{physics} \usepackage{mathtools}") # to enables physics extra latex package, can use \pdv for partial derivative 


class Intro(Scene): 
    def construct(self):
        arrowf = Arrow(start=LEFT*6, end=RIGHT*6).shift(UP)
        arrowb = Arrow(start=RIGHT*6, end=LEFT*6).shift(DOWN)

        f = Text("Forward Propagation \t \t \t... ", font="sans-serif", font_size=30).move_to(arrowf).shift(UP).shift(LEFT*3)
        f2 = Text("الانتشار  الامامي", font="sans-serif",font_size=30).move_to(arrowf).shift(UP).shift(RIGHT*4)
        b = Text("Backward Propagation\t \t \t...", font="sans-serif",  font_size=30).move_to(arrowb).shift(UP).shift(LEFT*3)
        b2 = Text("الانتشار  الخلفي", font="sans-serif", font_size=30).move_to(arrowb).shift(UP).shift(RIGHT*4)


        self.play(Write(arrowf),Write(f), Write(f2, reverse=True, remover=False))
        self.wait()
        self.play(Write(arrowb),Write(b), Write(b2, reverse=True, remover=False))
        self.wait()

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
        text2 = VGroup(input[1], hidden[1], output[1]).arrange(buff=2).to_edge(DOWN)
        

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
        self.play(AnimationGroup(Write(hidden_l1), Write(hidden_l2), Write(inputs_l1), Write(l1_l2), Write(l2_outputs), TransformMatchingShapes(text[1], text2[1], run_time=1)))
        self.wait()
        self.play(Transform(X, inputs),Transform(Y, outputs), Transform(inputs_l1, inputs_l1_v2),
                 Transform(l2_outputs, l2_outputs_v2),TransformMatchingShapes(text[0],text2[0]), TransformMatchingShapes(text[2],text2[2]))
        self.wait()


class Forward(Scene):
    def construct(self):
            # NN0 shallow 1 neuron
            network0 = NeuralNetworkMobject([1,1,1]).move_to(RIGHT*2.5)
            network0.label_inputs('x')
            network0.label_outputs('y')
            network0.label_hidden_layers('z')
            network0.scale(3)
            
            equation0 = MathTex(r"z_{1} &= wx_{1} + b  \\ a_{1} &= g(z_{1})", font_size=50).to_edge(LEFT)
            equation0[0][4:6].set_fill(color=BLUE_C)

            self.play(Write(network0))
            self.wait()
            self.play(AnimationGroup(Flash(network0[0][1], flash_radius=MED_LARGE_BUFF ,rate_func = rush_from, line_length=0.4, run_time=1.5), Write(equation0)))
            self.wait()

            # NN1 shallow 3 neurons 
            network1 = NeuralNetworkMobject([1,3,1]).move_to(RIGHT*2.5)
            network1.label_inputs('x')
            network1.label_outputs('y')
            network1.label_hidden_layers('z')

            network1.scale(3)

            equation1 = MathTex(r"z_{1} &= w_{1}x_{1} + b_{1}  \\ z_{2} &= w_{2}x_{1} + b_{2} \\ z_{3} &= w_{3}x_{1} + b_{3} \\ a_{1} &= g(z_{1}) \\ a_{2} &= g(z_{2}) \\ a_{3} &= g(z_{3})", font_size=50).to_edge(LEFT)
            equation1[0][5:7].set_fill(color=BLUE_C)
            equation1[0][15:17].set_fill(color=BLUE_C)
            equation1[0][25:27].set_fill(color=BLUE_C)

            self.play(Transform(network0,network1))
            self.wait()
            self.play(AnimationGroup(Flash(network1[0][1][0][0], flash_radius=MED_LARGE_BUFF ,rate_func = rush_from, line_length=0.3, run_time=1.5),
                                     Flash(network1[0][1][0][1], flash_radius=MED_LARGE_BUFF ,rate_func = rush_from, line_length=0.3, run_time=1.5),
                                     Flash(network1[0][1][0][2], flash_radius=MED_LARGE_BUFF ,rate_func = rush_from, line_length=0.3, run_time=1.5),
                                    TransformMatchingShapes(equation0,equation1, run_time=1)))
            self.wait()

            # NN2 deep 2l hidden 1 input 1 output
            network2 = NeuralNetworkMobject([1,3,2,1]).move_to(RIGHT*4)
            network2.label_inputs('x')
            network2.label_outputs('y')
            network2.label_hidden_layers('z')
            network2.scale(2)

            equation2 = MathTex(r"""z_{1}^{[2]} &= w_{1,1}^{[2]}a_{1}^{[1]} + w_{1,2}^{[2]}a_{2}^{[1]} + w_{1,3}^{[2]}a_{3}^{[1]} + b_{1}^{[2]}  \\ 
                                z_{2}^{[2]} &= w_{2,1}^{[2]}a_{1}^{[1]} + w_{2,2}^{[2]}a_{2}^{[1]} + w_{2,3}^{[2]}a_{3}^{[1]} + b_{2}^{[2]}  \\
                                a_{1}^{[2]} &= g(Z_{1}^{[2]}) \\ 
                                a_{2}^{[2]} &= g(Z_{2}^{[2]})""", font_size=40).to_edge(LEFT)

            layer0 = Tex(r"Input \\ $[l=0]$", font_size=25)
            layer1 = Tex(r"$[l=1]$", font_size=25)
            layer2 = Tex(r"$[l=2]$", font_size=25)
            layer3 = Tex(r"Output \\ $[l=3]$", font_size=25)
            layers = VGroup(layer0,layer1,layer2,layer3).arrange(buff=1).move_to(network2).shift(DOWN*2)

            self.play(Transform(network0,network2, run_time=2))
            self.wait()
            self.play(AnimationGroup(Flash(network2[0][2][0][0], flash_radius=MED_SMALL_BUFF ,rate_func = rush_from, line_length=0.25, run_time=1.5),
                                    Flash(network2[0][2][0][1], flash_radius=MED_SMALL_BUFF ,rate_func = rush_from, line_length=0.25, run_time=1.5),
                                    TransformMatchingShapes(equation1,equation2, run_time=2)))
            self.wait()
            self.play(Write(layers))
            self.play(Circumscribe(network2[0][1], run_time=2))
            self.wait()

class Notations(Scene):
    def construct(self):
        equation0 = MathTex(r"""z_{j, i}^{[l]} &= \sum_k w_{j, k}^{[l]} a_{k, i}^{[l - 1]} + b_j^{[l]}, \\
            a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}).""",
            font_size = 80, color=WHITE)
        
        for i in [2,12,19,29,35,43,49,61,73,77]:
            equation0[0][i].set_color(MAROON)
        for i in [4,14,31,37,45,63]:
            equation0[0][i].set_color(TEAL)
        for i in [6,25,39,53,65,80]:
            equation0[0][i].set_color(PURPLE)
        for i in [9,16,23]:
            equation0[0][i].set_color(YELLOW)
        for i in [75]:
            equation0[0][i].set_color(GOLD)                    

        self.play(Write(equation0))
        self.wait()
        self.play(ScaleInPlace(equation0, 0.5))
        self.play(equation0.animate.shift(UP*2.6))
        # self.play(AnimationGroup(ScaleInPlace(equation0, 0.35), equation0.animate.shift(LEFT*5)))
        self.wait()

        notation0 = Tex(r"$l = 1, \dots, L$ current layer. $l=0$ and $l=L$ are the input and output.", )
        notation1 = Tex(r"$n$ number of nodes in current layer.", )
        notation2 = Tex(r"$j = 1, \dots, n^{[l]}$ the $j$th node of current layer.", )
        notation3 = Tex(r"$k = 1, \dots, n^{[l-1]}$ the $k$th node of previous layer.", )
        notation4 = Tex(r"$i = 1, \dots, m$ current training example, where $m$ is number of training examples.", )
        notation5 = Tex(r"$g_j^{[l]}$ the activation function of current layer." )

        entity0 = MathTex(r"l", color=MAROON)
        entity1 = MathTex(r"n", color=GOLD)
        entity2 = MathTex(r"j", color=TEAL)
        entity3 = MathTex(r"k", color=YELLOW)
        entity4 = MathTex(r"i", color=PURPLE)
        entity5 = MathTex(r"g",)

        label0 = Tex('Entity')
        label1 = Tex('Description')

        table = MobjectTable([[entity0,notation0],
                        [entity1,notation1],
                         [entity2,notation2],
                          [entity3,notation3],
                           [entity4,notation4],
                            [entity5,notation5]], include_outer_lines=True, col_labels=[label0, label1],  line_config={"stroke_width": 1}).scale(0.53).move_to(DOWN*1.1)
        
        self.play(table.create())
        self.wait()

class Cost_for(Scene):
    def construct(self):
        eq0 = MathTex(r"a^{[L]} =\hat{y}")
        eq1 = MathTex(r"J = f(\hat{y}, y)")
        eq = VGroup(eq0,eq1).arrange(DOWN).scale(3)

        self.play(Write(eq))
        self.wait()

class Costs(Scene):
    def construct(self):

        title = Title("Cross-entropy Loss")
        # Binary Classification
        rect1 = Rectangle(height=3*1.05, width=4*1.05, color=WHITE, fill_opacity=0, stroke_width=0.5).to_edge(LEFT)
        eq1 = MathTex(r"""
        J &= f({\hat{y}}, {y}) = f({a}^{[L]}, {y}) \\
        &= -\frac{1}{m} \sum_i (y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)) \\
        &= -\frac{1}{m} \sum_i (y_i \log(a_i^{[L]}) + (1 - y_i) \log(1 - a_i^{[L]})).
        """, font_size=15).move_to(rect1)
        label1 = Tex(r"Binary Classification", color=WHITE,  font_size=32).move_to(rect1, DOWN).shift(DOWN)

        # Multiclass Classification
        rect2 = Rectangle(height=3*1.05, width=4*1.05, color=WHITE, fill_opacity=0, stroke_width=0.5)
        eq2 = MathTex(r"""
        J &= f({\hat{y}}, {y}) = f({a}^{[L]}, {y}) \\
        &= -\frac{1}{m} \sum_i \sum_j y_{j, i} \log(\hat{y}_{j, i}) \\
        &= -\frac{1}{m} \sum_i \sum_j y_{j, i} \log(a_{j, i}^{[L]}).
        """, font_size=20).move_to(rect2)
        label2 = Tex(r"Multiclass Classification", color=WHITE,  font_size=32).move_to(rect2, DOWN).shift(DOWN)

        # Multi-Label Classification
        rect3 = Rectangle(height=3*1.05, width=4*1.05, color=WHITE, fill_opacity=0, stroke_width=0.5).to_edge(RIGHT)
        eq3 = MathTex(r"""
        J &= f({\hat{y}}, {y}) = f({a}^{[L]}, {y}) \\
        &= \sum_j \Bigl(-\frac{1}{m} \sum_i (y_{j, i} \log(\hat{y}_{j, i}) + (1 - y_{j, i}) \log(1 - \hat{y}_{j, i}))\Bigr) \\
        &= \sum_j \Bigl(-\frac{1}{m} \sum_i (y_{j, i} \log(a_{j, i}^{[L]}) + (1 - y_{j, i}) \log(1 - a_{j, i}^{[L]}))\Bigr),
        """, font_size=15).move_to(rect3)       
        label3 = Tex(r"Multi-Label Classification", color=WHITE,  font_size=32).move_to(rect3, DOWN).shift(DOWN)

        self.play(Write(title))
        self.play(AnimationGroup(Create(rect1),Create(rect2),Create(rect3)))
        self.play(AnimationGroup(Write(label1),Write(label2),Write(label3)))
        self.play(AnimationGroup(Write(eq1), Write(eq2), Write(eq3)))
        self.wait()

        ## GRADIENTS 
        title2 = Title("Derivative (gradients) of Cross-entropy Loss")
        # Binary Classification
        eq12 = MathTex(r"""
            \pdv{J}{a_i^{[L]}} = \frac{1}{m} \Bigl(\frac{1 - y_i}{1 - a_i^{[L]}} - \frac{y_i}{a_i^{[L]}}\Bigr)
            """, font_size=15*2).move_to(rect1)

        # Multiclass Classification
        eq22 = MathTex(r"""
            \pdv{J}{a_{j, i}^{[L]}} = -\frac{1}{m} \frac{y_{j, i}}{a_{j, i}^{[L]}}
            """, font_size=20*2).move_to(rect2)

        # Multi-Label Classification
        eq32 = MathTex(r"""
            \pdv{J}{a_{j, i}^{[L]}} = \frac{1}{m} \Bigl(\frac{1 - y_{j, i}}{1 - a_{j, i}^{[L]}} - \frac{y_{j, i}}{a_{j, i}^{[L]}}\Bigr)
            """, font_size=15*2).move_to(rect3)     

        self.play(Transform(title,title2))
        self.play(AnimationGroup(Transform(eq1,eq12),Transform(eq2,eq22), Transform(eq3, eq32)))
        self.wait()

class Act(Scene):
    def construct(self):
        title = Title("Activation Functions")
        # sigmoid
        rect1 = Rectangle(height=3*1.05, width=4*1.05, color=WHITE, fill_opacity=0, stroke_width=0.5).to_edge(LEFT)
        eq1 = MathTex(r"""
            a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
            &= \sigma(z_{j, i}^{[l]}) \\
            &= \frac{1}{1 + \exp(-z_{j, i}^{[l]})}
        """, font_size=20).move_to(rect1)
        label1 = Tex(r"Sigmoid Activation", color=WHITE,  font_size=32).move_to(rect1, DOWN).shift(DOWN)

        # ReLU
        rect2 = Rectangle(height=3*1.05, width=4*1.05, color=WHITE, fill_opacity=0, stroke_width=0.5)
        eq2 = MathTex(r"""
            a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
            &= \max(0, z_{j, i}^{[l]}) \\
            &=
            \begin{cases}
            z_{j, i}^{[l]} &\text{if } z_{j, i}^{[l]} > 0, \\
            0 &\text{otherwise}
            \end{cases}
        """, font_size=23).move_to(rect2)
        label2 = Tex(r"ReLU Activation", color=WHITE,  font_size=32).move_to(rect2, DOWN).shift(DOWN)

        # softmax
        rect3 = Rectangle(height=3*1.05, width=4*1.05, color=WHITE, fill_opacity=0, stroke_width=0.5).to_edge(RIGHT)
        eq3 = MathTex(r"""
            a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
            &= \frac{\exp(z_{j, i}^{[l]})}{\sum_p \exp(z_{p, i}^{[l]})}
        """, font_size=20).move_to(rect3)       
        label3 = Tex(r"Softmax Activation", color=WHITE,  font_size=32).move_to(rect3, DOWN).shift(DOWN)

        self.play(Write(title))
        self.play(AnimationGroup(Create(rect1),Create(rect2),Create(rect3)))
        self.play(AnimationGroup(Write(label1),Write(label2),Write(label3)))
        self.play(AnimationGroup(Write(eq1), Write(eq2), Write(eq3)))
        self.wait()

        ## GRADIENTS 
        title2 = Title("Derivative (gradients) of Activation Functions")
        # sigmoid
        eq12 = MathTex(r"""
            \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} &=a_{j, i}^{[l]} (1 - a_{j, i}^{[l]}) \\
            \pdv{a_{p, i}^{[l]}}{z_{j, i}^{[l]}} = 0, \quad \forall p \ne j
            """, font_size=20*1.5).move_to(rect1)

        # relu
        eq22 = MathTex(r"""
            \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} &\coloneqq
            \begin{cases}
            1 &\text{if } z_{j, i}^{[l]} > 0, \\
            0 &\text{otherwise},
            \end{cases} \\
            \pdv{a_{p, i}^{[l]}}{z_{j, i}^{[l]}} &= 0, \quad \forall p \ne j
            """, font_size=20*1.5).move_to(rect2)

        # softmax
        eq32 = MathTex(r"""
            \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} &= a_{j, i}^{[l]} (1 - a_{j, i}^{[l]}), \notag \\
            \pdv{a_{p, i}^{[l]}}{z_{j, i}^{[l]}} &= -a_{p, i}^{[l]} a_{j, i}^{[l]}, \quad \forall p \ne j. \notag
            """, font_size=18*1.5).move_to(rect3)     

        self.play(Transform(title,title2))
        self.play(AnimationGroup(Transform(eq1,eq12),Transform(eq2,eq22), Transform(eq3, eq32)))
        self.wait()

class Grad_des(Scene):
    def construct(self):
        
        # Minimum of a Function
        title0 = Title("Minimum of a Function")
        ax0 = Axes(x_range=[-4,2], y_range=[-1,8], tips=False, stroke_width=1, axis_config={'color': WHITE}).scale(0.7).shift(DOWN*0.8).shift(RIGHT*2.5)
        graph0 = ax0.plot(lambda x: x**4 +3*x**3 + 0.1*x**2 - 2.3*x+ 4, color=BLUE, x_range=[-2.855,1.162])
        graph0_d = ax0.plot(lambda x: 4*x**3+9*x**2+0.2*x-2.3)
        x_label = ax0.get_x_axis_label("x")

        eq0 = MathTex(r"f'(x) &= 0 \quad \text{solve for } x, \\ x_{1} &= 0.452, \\ x_{2} &= -2.095", font_size=40).to_corner(LEFT)
        eq0[0][17:19].set_color(RED)
        eq0[0][26:28].set_color(RED)
        dot0 = Dot(color=RED).move_to(ax0.c2p(-2.095, 0.936,0))
        v_line0 = ax0.get_vertical_line(ax0.c2p(-2.095, 0.936,0), line_config={"dashed_ratio": 0.70})
        dot1 = Dot(color=RED).move_to(ax0.c2p(0.452, 3.3,0))
        v_line1 = ax0.get_vertical_line(ax0.c2p(0.452, 3.3,0), line_config={"dashed_ratio": 0.70})

        self.play(LaggedStart(DrawBorderThenFill(ax0),Create(graph0),Write(title0), Write(x_label), run_time=3, lag_ratio=0.5))
        self.play(Write(eq0, run_time=3))
        self.play(Write(VGroup(dot0, dot1, v_line0, v_line1)))
        self.wait()

        # Gradient decent

        title1 = Title("Gradient Descent")
        eq1 = MathTex(r"J &= f(\hat{y}, y), \quad a^{[L]} = \hat{y} \\ a^{[L]} &= g(z^{[L]}), \quad z = h(w^{[L]},b^{[L]},a^{[L-1]}) \\  a^{[L-1]} &= g(z^{[L-1]}), \quad z = h(w^{[L-1]},b^{[L-1]},a^{[L-2]}) \\ a^{[L-2]} &= \dots", font_size=28).to_corner(LEFT)
        eq2_0 = Tex(r"$J(x)$",font_size=50)
        eq2_1 = Tex(r"$x_{\text{new}} \coloneqq x_{\text{old}} - \dv{J(x)}{x_{\text{old}}} * \alpha$",font_size=50)
        eq2 = VGroup(eq2_0,eq2_1).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(LEFT)
        value = ValueTracker(1)
        
        moving_slope = always_redraw(
            lambda: ax0.get_secant_slope_group(
                x=value.get_value(),
                graph=graph0,
                dx=0.005,
                dx_line_color=YELLOW,
                secant_line_length=4,
                secant_line_color=YELLOW,
                include_secant_line=True,             
            )[2].set_stroke(width=3)
        )

        dot2 = always_redraw(
            lambda: Dot(color=RED).move_to(
            ax0.c2p(value.get_value(), graph0.underlying_function(value.get_value()))))
        
        v_line = always_redraw(
            lambda:
            ax0.get_vertical_line(ax0.c2p(
            value.get_value(), graph0.underlying_function(value.get_value())), line_config={"dashed_ratio": 0.70}))
        
        arrow = always_redraw(
            lambda: Arrow(buff=0, 
                        start=v_line.get_center(),
                        end=(v_line.get_center() + (graph0_d.underlying_function(value.get_value()) * -0.1, 0, 0)),
                        max_tip_length_to_length_ratio=0.1,
                        max_stroke_width_to_length_ratio=2)
        )
        
        slope_value_text = (
            Tex(r"$\dv{J(x)}{x}$ (Slope): ", font_size=40)
            .next_to(ax0, DOWN, buff=0.1)
            .set_color(YELLOW)
            .add_background_rectangle()
        )

        slope_value = always_redraw(
            lambda: DecimalNumber(num_decimal_places=1, font_size=40)
            .set_value(graph0_d.underlying_function(value.get_value()))
            .next_to(slope_value_text, RIGHT, buff=0.1)
            .set_color(YELLOW)
        )

        self.play(FadeOut(eq0,dot0,dot1,v_line0, v_line1))
        self.play(Write(eq1))
        self.wait()
        
        self.play(FadeOut(eq1))
        self.play(AnimationGroup(Write(eq2), Transform(title0,title1)))
        self.play(Write(VGroup(dot2, moving_slope,slope_value_text, slope_value, v_line, arrow)))
        
        # Steps
        def powspace(start, stop, power, num):
            start = np.power(start, 1/float(power))
            stop = np.power(stop, 1/float(power))
            return np.power( np.linspace(start, stop, num=num), power) 

        steps1 = powspace(start=1,stop=0.452, power=20000, num=10)
        #steps1 = np.linspace(1,0.452, num=10)

        for i in steps1:
            self.play(value.animate.set_value(i), run_time=0.8, rate_functions=rate_functions.linear)

        self.wait()
        value.set_value(-2.8)

        steps2 = powspace(start=2.8,stop=2.095, power=20000, num=10) * - 1.0
        #steps2 = np.linspace(-2.8,-2.095, num=10)
        for i in steps2:
            self.play(value.animate.set_value(i), run_time=0.8, rate_functions=rate_functions.linear)

        # continous
        self.wait()
        value.set_value(1)
        self.play(value.animate.set_value(0.452), run_time=5, rate_functions=rate_functions.ease_out_sine)
        #self.play(FadeOut(dot2,moving_slope,slope_value))
        self.wait()
        value.set_value(-2.8)
        #self.play(FadeIn(dot2,moving_slope,slope_value))
        self.play(value.animate.set_value(-2.095), run_time=5, rate_functions=rate_functions.ease_out_sine)
        self.wait()

class Back(Scene):
    def construct(self):
        title = Title(r"$J =  f(a^{[L]}, y), \quad a^{[l]} = g(z^{[l]}),  \quad z^{[l]} = w^{[l]} * a^{[l-1]}+b^{[l]}$").to_edge(UP)

        eq0 = MathTex(r"""\pdv{J}{w_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} \pdv{z_{j, i}^{[l]}}{w_{j, k}^{[l]}} \\ 
                      \pdv{J}{w_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} \pdv{}{w_{j, k}^{[l]}} \left( w_{j, k}^{[l]} a_{k, i}^{[l - 1]} + b_j^{[l]} \right) \\
                      \pdv{J}{w_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} a_{k, i}^{[l - 1]}
                      """, font_size= 33).to_corner(LEFT).shift(DOWN)
        
        eq1 = MathTex(r"""\pdv{J}{b_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} \pdv{z_{j, i}^{[l]}}{b_{j, k}^{[l]}} \\ 
                      \pdv{J}{b_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} \pdv{}{w_{j, k}^{[l]}} \left( w_{j, k}^{[l]} a_{k, i}^{[l - 1]} + b_j^{[l]} \right) \\
                      \pdv{J}{b_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}}""", font_size= 33).to_corner(RIGHT).shift(DOWN)


        self.play(Write(title)) 
        self.play((Write(eq0, run_time =5)))
        self.wait()
        self.play((Write(eq1, run_time =5)))
        self.wait()

        eq0_1 = MathTex(r"\pdv{J}{w_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} a_{k, i}^{[l - 1]}", font_size= 50).to_corner(LEFT).shift(UP)
        eq1_1 = MathTex(r"\pdv{J}{b_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}}", font_size= 50).to_corner(RIGHT).shift(UP)
        eq3 = MathTex(r"\pdv{J}{a_{k, i}^{[l - 1]}} = \sum_j \pdv{J}{z_{j, i}^{[l]}} w_{j, k}^{[l]}", font_size=50).shift(DOWN*2)


        self.play(Transform(eq0,eq0_1))
        self.play(Transform(eq1,eq1_1))
        self.play(FadeIn(eq3))
        self.wait()
        self.play(eq0_1[0][14:25].animate.set_color(YELLOW),eq1_1[0][14:].animate.set_color(YELLOW), eq3[0][16:27].animate.set_color(YELLOW))
        self.wait()
        #self.add(index_labels(eq0_1[0]), index_labels(eq1_1[0]))
        




        eq2 = MathTex(r"\pdv{J}{z_{j, i}^{[l]}} = \sum_j \pdv{J}{a_{j, i}^{[l]}} \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}}", font_size = 80)
        eq2[0][0:11].set_color(YELLOW)

        self.play(TransformMatchingShapes(VGroup(eq0, eq1, eq0_1, eq1_1,eq3),eq2))
        self.wait()
        self.play(eq2.animate.set_color(WHITE))
        self.play(Indicate(eq2[0][14:25]))
        self.wait()
        self.play(Indicate(eq2[0][25:]))
        #elf.add(index_labels(eq2[0]))
        self.wait()

class Summary(Scene):
    def construct(self):

        input_C0 = Circle(radius=0.2, color=BLUE_C, stroke_color=BLACK, fill_opacity=1)
        hidden_C0 = Circle(radius=0.2, color=WHITE, stroke_color=BLACK, fill_opacity=1)
        output_C0 = Circle(radius=0.2, color=RED_C, stroke_color=BLACK, fill_opacity=1)
        nn = VGroup(input_C0, hidden_C0, output_C0).arrange(buff=5).to_edge(UP)

        arrow1 = Arrow(start=input_C0, end=hidden_C0, max_tip_length_to_length_ratio=0.05)
        arrow2 = Arrow(start=hidden_C0, end=output_C0, max_tip_length_to_length_ratio=0.05)
        arrows = VGroup(arrow1, arrow2)

        self.play(Write(nn), Create(arrows), run_time=2)
        self.wait()

        f1 = Arrow(start=input_C0, end=output_C0, max_tip_length_to_length_ratio=0.02, max_stroke_width_to_length_ratio=0.3).shift(DOWN*1.5).shift((0,0.5,0))
        f1_label = Tex(r"Forward Propagation", font_size=40).next_to(hidden_C0, DOWN*3).shift((0,0.5,0))
        i1 = MathTex(r"x", color=BLUE, font_size=80).next_to(input_C0, DOWN*6).shift((0,0.5,0))
        h1 = MathTex(r"z^{[1]} &= w^{[1]}x^{[0]}+b^{[1]} \\ a^{[1]} &= g(z^{[1]})", font_size=25).next_to(hidden_C0, DOWN*6).shift((0,0.5,0))
        o1 = MathTex(r"z^{[2]} &= w^{[2]}a^{[1]}+b^{[2]} \\ a^{[2]} &= g(z^{[2]}) \\ \text{loss} &= J(a^{[2]},y)", font_size=20).next_to(output_C0, DOWN*6).shift((0,0.5,0))
        o1[0][-2].set_color(RED)
        h1[0][9].set_color(BLUE)

        self.play(Create(f1), Write(f1_label))
        self.wait()
        self.play(Write(i1))
        self.play(Write(h1))
        self.play(Write(o1))
        self.wait()

        f2 = Arrow(start=output_C0, end=input_C0, max_tip_length_to_length_ratio=0.02, max_stroke_width_to_length_ratio=0.3).shift(DOWN*1.5 - (0,2,0)).shift((0,0.5,0))
        f2_label = Tex(r"Backward Propagation", font_size=40).next_to(hidden_C0, DOWN).shift(DOWN - (0,1.5,0)).shift((0,0.5,0))
        i2 = MathTex(r"x", color=BLUE, font_size=80).next_to(input_C0, DOWN).shift(DOWN*1.5 - (0,2,0)).shift((0,0.7,0))
        h2 = MathTex(r""" \pdv{a^{[1]}}{z^{[1]}} &= g'(z^{[1]}) \\ 
            \pdv{J}{z^{[1]}} &= \pdv{J}{a^{[1]}} \pdv{a^{[1]}}{z^{[1]}} \\
            \pdv{J}{w^{[1]}} &= \pdv{J}{z^{[1]}} x^{[0]} \\
            \pdv{J}{b^{[1]}} &= \pdv{J}{z^{[1]}}
            """, font_size=22).next_to(hidden_C0, DOWN).shift(DOWN*1.5 - (0,2,0)).shift((0,0.7,0))
        o2 = MathTex(r"""\pdv{J}{a^{[2]}} &= J'(a^{[2]},y), \hspace{0.3cm}  \pdv{a^{[2]}}{z^{[2]}} = g'(z^{[2]}) \\
            \pdv{J}{z^{[2]}} &= \pdv{J}{a^{[2]}} \pdv{a^{[2]}}{z^{[2]}} \\
            \pdv{J}{w^{[2]}} &= \pdv{J}{z^{[2]}} a^{[1]} \\ 
            \pdv{J}{b^{[2]}} &= \pdv{J}{z^{[2]}} \\
            \pdv{J}{a^{[1]}} &= \pdv{J}{z^{[2]}} w^{[2]}
            """, font_size=20).next_to(output_C0, DOWN).shift(DOWN*1.5 - (0,2,0)).shift((0,0.7,0))
  
        h2[0][65].set_color(BLUE)
        o2[0][17].set_color(RED)

        update = Tex(r"Gradient Descent: parameter$_{\text{new}}$ $\coloneqq$ parameter$_{\text{old}}$ - $\nabla$ parameter$_{\text{old}} * \alpha $", font_size=30).to_edge(DOWN)
        self.play(Create(f2), Write(f2_label, reverse=True))
        self.add(f2_label)
        self.wait()
        self.play(Write(o2, run_time=4))
        self.play(Write(h2, run_time=3))
        self.play(Write(i2))
        self.wait()
        self.play(Write(update))
        #self.add(index_labels(h1[0]), index_labels(h2[0]))
        self.wait()
 
class Test(Scene): #Scrapped 
    def construct(self):
        mynetwork = NeuralNetworkMobject([1,3,5,2,1])
        mynetwork.label_inputs('x')
        mynetwork.label_outputs('y')
        mynetwork.label_hidden_layers('z')
        mynetwork.label_outputs_text(['Number'])
        mynetwork.scale(3)
        self.play(Write(mynetwork))
        self.wait

# A customizable Sequential Neural Network, copied from https://www.youtube.com/watch?v=HnIeAP--vWc and adjusted for manim community version
class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": MED_LARGE_BUFF,

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
        "arrow": True,
        "arrow_tip_size": 0.08,
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
            dots = MathTex(r"\\vdots")
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