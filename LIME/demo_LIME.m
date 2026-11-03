demoDir = fileparts(mfilename('fullpath'));
if isempty(demoDir)
    error('LIME:ScriptPath', '请先将本文件保存为 demo_LIME.m 再运行。');
end
projectDir = fileparts(demoDir);

inputDir = fullfile(projectDir, 'data', 'eval', 'low');
referenceDir = fullfile(projectDir, 'data', 'eval', 'high');
outputDir = fullfile(demoDir, 'results_lol');
expectedImageCount = 15;

evaluateExistingResults = true;
alpha = 0.15;
gamma = 0.8;
sigma = 2;
epsilon = 1e-3;

runLimeDemo(demoDir, inputDir, referenceDir, outputDir, expectedImageCount, ...
    evaluateExistingResults, alpha, gamma, sigma, epsilon);

function runLimeDemo(demoDir, inputDir, referenceDir, outputDir, expectedCount, ...
        evaluateExistingResults, alpha, gamma, sigma, epsilon)
    if ~isfolder(inputDir)
        error('LIME:MissingInput', '找不到测试输入目录：%s', inputDir);
    end
    if ~isfolder(referenceDir)
        error('LIME:MissingReference', '找不到正常曝光参考目录：%s', referenceDir);
    end
    coreFile = fullfile(demoDir, 'lime_enhance.m');
    if ~isfile(coreFile)
        error('LIME:MissingCore', '缺少核心函数：%s', coreFile);
    end

    files = dir(fullfile(inputDir, '*.png'));
    files = files(~[files.isdir]);
    [~, order] = sort({files.name});
    files = files(order);
    if isempty(files)
        error('LIME:EmptyInput', '输入目录中没有 PNG 图像：%s', inputDir);
    end
    if numel(files) ~= expectedCount
        error('LIME:ImageCount', ...
            '预期 %d 张测试图，实际 %d 张。请核对 inputDir 和 expectedImageCount。', ...
            expectedCount, numel(files));
    end
    if isfile(outputDir)
        error('LIME:OutputIsFile', '输出路径已是文件：%s', outputDir);
    end

    outputExists = false(numel(files), 1);
    for index = 1:numel(files)
        outputFile = fullfile(outputDir, files(index).name);
        if isfolder(outputFile)
            error('LIME:OutputIsFolder', '预期的输出文件位置已是目录：%s', outputFile);
        end
        outputExists(index) = isfile(outputFile);
        info = imfinfo(fullfile(inputDir, files(index).name));
        if numel(info) ~= 1 || ~strcmp(info.ColorType, 'truecolor')
            error('LIME:InputFormat', '需要单帧 RGB PNG：%s', files(index).name);
        end
        referenceFile = fullfile(referenceDir, files(index).name);
        if ~isfile(referenceFile)
            error('LIME:MissingPair', '缺少同名参考图片：%s', referenceFile);
        end
        validateComparisonImage(referenceFile, info.Height, info.Width);
        if outputExists(index)
            validateComparisonImage(outputFile, info.Height, info.Width);
        end
    end
    reuseResults = all(outputExists) && evaluateExistingResults;
    if any(outputExists) && ~reuseResults
        error('LIME:OutputExists', ...
            ['已存在 %d/%d 张输出图片。完整结果可设置 evaluateExistingResults=true 进行评估；' ...
            '\n重新增强或已有结果不完整时，请将 outputDir 改为新的目录。'], ...
            nnz(outputExists), numel(files));
    end

    originalPath = path;
    restorePath = onCleanup(@() path(originalPath));
    addpath(demoDir, '-begin');
    if ~isfolder(outputDir)
        [ok, message] = mkdir(outputDir);
        if ~ok
            error('LIME:CreateOutput', '无法创建输出目录：%s\n%s', outputDir, message);
        end
    end

    fprintf('LIME 论文复现：sped-up solver，无训练，无 BM3D。\n');
    fprintf('输入：%s\n参考：%s\n输出：%s\n图像数量：%d\n', ...
        inputDir, referenceDir, outputDir, numel(files));
    if reuseResults
        fprintf('检测到完整的已有结果：仅评估这些 PNG，本次不执行增强或应用增强参数。\n');
    else
        fprintf('alpha=%g, gamma=%g, sigma=%g, epsilon=%g\n', ...
            alpha, gamma, sigma, epsilon);
    end
    fprintf('误差范围：RGB 全通道，归一化像素值 [0,1]；最终对各图指标等权平均。\n');
    maeValues = zeros(numel(files), 1);
    mseValues = zeros(numel(files), 1);
    totalTimer = tic;
    for index = 1:numel(files)
        imageTimer = tic;
        outputFile = fullfile(outputDir, files(index).name);
        if ~reuseResults
            inputFile = fullfile(inputDir, files(index).name);
            rgb = readRGBNormalized(inputFile);
            enhanced = lime_enhance(rgb, 'Alpha', alpha, 'Gamma', gamma, ...
                'Sigma', sigma, 'Epsilon', epsilon);
            imwrite(uint8(round(255 * enhanced)), outputFile);
        end

        outputPixels = readRGBNormalized(outputFile);
        referencePixels = readRGBNormalized(fullfile(referenceDir, files(index).name));
        difference = outputPixels - referencePixels;
        maeValues(index) = mean(abs(difference(:)));
        mseValues(index) = mean(difference(:) .^ 2);
        fprintf('[%d/%d] %s：MAE=%.8f，MSE=%.8f，%.3f 秒\n', ...
            index, numel(files), files(index).name, ...
            maeValues(index), mseValues(index), toc(imageTimer));
    end
    fprintf('\n完成评估：%d 张图片，总耗时 %.3f 秒。\n', ...
        numel(files), toc(totalTimer));
    fprintf('平均绝对误差 MAE（[0,1] 归一化像素尺度）：%.8f\n', mean(maeValues));
    fprintf('平均均方误差 MSE（归一化像素尺度的平方）：%.8f\n', mean(mseValues));
    fprintf('结果目录：%s\n', outputDir);
end

function validateComparisonImage(filePath, expectedHeight, expectedWidth)
    info = imfinfo(filePath);
    if numel(info) ~= 1 || ~strcmp(info.ColorType, 'truecolor')
        error('LIME:ComparisonFormat', '需要单帧 RGB PNG：%s', filePath);
    end
    if info.Height ~= expectedHeight || info.Width ~= expectedWidth
        error('LIME:SizeMismatch', '图片尺寸与低光输入不一致：%s', filePath);
    end
end

function pixels = readRGBNormalized(filePath)
    image = imread(filePath);
    if ndims(image) ~= 3 || size(image, 3) ~= 3 || ...
            (~isa(image, 'uint8') && ~isa(image, 'uint16'))
        error('LIME:ImageType', '需要 uint8 或 uint16 RGB PNG：%s', filePath);
    end

    pixels = double(image) / double(intmax(class(image)));
end
