@testset "StateBlockLayout" begin
    layout = StateBlockLayout(:juvenile => 3, :adult => 2, :seedbank => 1)

    @test blocknames(layout) == [:juvenile, :adult, :seedbank]
    @test length(layout) == 6
    @test blockrange(layout, :juvenile) == 1:3
    @test layout[:adult] == 4:5
    @test blockranges(layout)[:seedbank] == 6:6

    state = [1.0, 2.0, 3.0, 10.0, 11.0, 100.0]
    blocks = split_state(state, layout)
    @test blocks[:juvenile] == [1.0, 2.0, 3.0]
    @test blocks[:adult] == [10.0, 11.0]
    @test blocks[:seedbank] == [100.0]

    views = split_state(state, layout; copy=false)
    views[:juvenile][1] = 9.0
    @test state[1] == 9.0

    recombined = combine_state(layout, Dict(
        :juvenile => [9.0, 2.0, 3.0],
        :adult => [10.0, 11.0],
        :seedbank => [100.0],
    ))
    @test recombined == state

    @test_throws DimensionMismatch StateBlockLayout([:a, :b], [2])
    @test_throws ArgumentError StateBlockLayout(:a => 2, :a => 1)
    @test_throws ArgumentError StateBlockLayout(:a => 0)
    @test_throws DimensionMismatch split_state([1.0, 2.0], layout)
    @test_throws DimensionMismatch combine_state(layout, Dict(
        :juvenile => [1.0, 2.0],
        :adult => [3.0, 4.0],
        :seedbank => [5.0],
    ))
end
