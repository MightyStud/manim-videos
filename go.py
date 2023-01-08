from manim import *
import math 

config.background_color = WHITE

class nneq(Scene):
    def construct(self):
        eq = MathTex(r"WX + b", font_size=350)
        eq.set_color(BLACK)
        self.play(Write(eq, run_time=3.5))


class Function(Scene):
    def construct(self):
        ax = Axes(               
            x_range=[-10,10],
            y_range=[-7,7],
            tips=False,
            axis_config={
                'color': BLACK,
            }
        )
        graph1 = ax.plot(lambda x: x, x_range=[-10,10], color=ORANGE)
        graph2 = ax.plot(lambda x: x*x, x_range=[-10,10], color=ORANGE)
        graph3 = ax.plot(lambda x: math.sin(x), x_range=[-10,10], color=ORANGE)

        label1 = MathTex(r"y = f(x) = x", font_size=50, color=ORANGE).to_edge(UL)
        label2 = MathTex(r"y = f(x) = x^2", font_size=50, color=ORANGE).to_edge(UL)
        label3 = MathTex(r"y = f(x) = sin(x)", font_size=50, color=ORANGE).to_edge(UL)


        self.play(Create(ax))
        self.play(AnimationGroup(Write(graph1, run_time=2), Write(label1, run_time=1)))
        self.play(AnimationGroup(Transform(graph1, graph2, run_time=2), Transform(label1, label2, run_time=1)))
        self.play(AnimationGroup(Transform(graph1, graph3, run_time=2), Transform(label1, label3, run_time=1)))
        self.wait(1)
        self.play(AnimationGroup(FadeOut(graph1, run_time=2), FadeOut(label1, run_time=2)))





