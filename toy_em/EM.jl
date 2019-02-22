mutable struct EM
    probA::Float32
    probB::Float32
    probC::Float32
end

EM() = EM(0.5, 0.5, 0.5)
"""
    E_step(self::EM, iter::Int32) -> Float32

返回一个浮点数μ^{i+1}
"""
function E_step(self::EM, data::Bool)
    probFromB = self.probA * self.probB^data * (1-self.probB)^(1-data)
    probSum = probFromB + 
                (1-self.probA) * self.probC^data * (1-self.probC)^(1-data)
    return probFromB / probSum
end

function M_step(self::EM, datas::Array{Bool, 1}; maxIter=4)
    for i in 1:maxIter
    muProbs = [E_step(self, data) for data in datas]
    self.probA = 1/length(datas) * sum(muProbs)
    self.probB = sum([mu*data for (mu, data) in zip(muProbs, datas)]) / sum(muProbs)
    self.probC = sum([(1-mu)*data for (mu, data) in zip(muProbs, datas)]) / sum(map(mu -> 1-mu, muProbs))
    println("第$(i)轮结果:")
    println("A的概率为$(self.probA)")
    println("B的概率为$(self.probB)")
    println("C的概率为$(self.probC)\n")
    end
end

function EMbase()
    boolArray(x) = map(Bool, x)
    datas = boolArray([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    #M_step(EM(), datas)
    M_step(EM(0.4, 0.6, 0.7), datas)
end

mutable struct GaussModel
    alpha::Float32
    mu::Float32
    sigma::Float32
end

gaussProb(gm::GaussModel, x::AbstractFloat) = (1/sqrt(gm.sigma)) * exp(-(x-gm.mu)^2 / (2gm.sigma^2))

gausses = [GaussModel(0.3, 1.2, 2), GaussModel(0.5, 1., 0.4), GaussModel(0.2, 0.8, 1.3)]

using Distributions
"""
    GMMgenerator() -> Float32
产生基于gausses的数据
"""
function GMMgenerator()
    randNum, accumulate = rand(), 0.
    for (i, gauss) in enumerate(gausses)
        if randNum > accumulate
            return rand(Normal(gauss.mu, gauss.sigma))
        else
            accumulate += gauss.alpha
        end
    end
end


function GMM(_K=3, maxIter=10)
    _K = length(gausses)
    println("生成数据的Gauss分布为：")
    println(gausses, "\n")
    # Init
    datas = [GMMgenerator() for _ in 1:30000]
    alphas = [rand() for _ in 1:_K]; alphas ./= sum(alphas)
    fakeGausses = [GaussModel(alphas[i], rand(), rand()) for i in 1:_K]
    
    # 判断循环终止条件
    for _ in 1:maxIter
    println(fakeGausses)

    # E_step
    response = zeros(_K, length(datas))
    for j in 1:length(datas)
        dominators = zeros(1, _K)
        for k in 1:_K
            response[k, j] = gaussProb(fakeGausses[k], datas[j])
        response[:, j] ./= sum(response[:, j])
        end
    end

    # M_step
    for k in 1:_K
        old_mu = fakeGausses[k].mu
        fakeGausses[k].mu   = sum( map(x -> x[1]*x[2] ,zip(response[k, :], datas)) ) / sum(response[k, :])
        fakeGausses[k].sigma= sum( map(x -> x[1]*(x[2]-old_mu)^2, zip(response[k, :], datas)) ) / sum(response[k, :])
        fakeGausses[k].alpha= sum( response[k, :] ) / length(datas)
    end

    end

    println("GaussMixtureModel运行完毕")
    println("共$(_K)个高斯分布")
    for k in 1:_K
        println("第$(k)个高斯分布参数：")
        println("权重：$(fakeGausses[k].alpha), 均值：$(fakeGausses[k].mu), 方差：$(fakeGausses[k].sigma)")
    end

end


function main()
    GMM()
end

main()
