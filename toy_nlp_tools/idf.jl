struct FileHolder
    fileProcessor::Dict{String, Function}

    FileHolder(processor::Dict{String, Function}) = new(processor)
end

function FileHolder(filenames::Vector{String})
    processor(line::String) = split(line, ' ')
    FileHolder(Dict{String, Function}(filename => processor) for filename in filenames)
end

struct OutSetting
    outFilename::String
    outFileFormat::Symbol
end

__idfCaculator(value, documentCount) = log(documentCount / (value != 0 ? value : 1)

mutable struct IDF
    idfDict::Dict{String, Int64}
    idfCalculator::Function
    documentCount::Int64

    fileHolder::FileHolder
    outSetting::OutSetting
    IDF(fileHolder::FileHolder， outSetting::OutSetting("out.txt"); idfCalculator::Function=__idfCaculator) = 
        new(Dict{String, Int64}, idfCalculator, 0, fileHolder, outSetting)
end

function fit!(self::IDF)
    # TODO: parallize it
    for (filename, processor) in self.fileHolder
        open(filename) do file
            statisticIDF!(self, file, processor)
        end
    end
end

function statisticIDF!(self::IDF, file, processor::Function)
    for line in eachline(file)
        tokens = processor(line)
        for token in Set(tokens)
            self.idfDict[token] = get(self.idfDict, token, 0) + 1
        end
        self.documentCount += 1
    end
end

function transform()

end

getIdfValue(self::IDF, key::String) = self.caculator(get(self.idfDict, key, 0), documentCount)