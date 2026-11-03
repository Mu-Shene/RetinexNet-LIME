function [enhanced, illumination] = lime_enhance(rgb, varargin)

    validateattributes(rgb, {'single', 'double'}, ...
        {'real', 'finite', 'nonempty', '>=', 0, '<=', 1}, ...
        mfilename, 'rgb', 1);
    if ndims(rgb) ~= 3 || size(rgb, 3) ~= 3
        error('LIME:InvalidRGB', '输入必须为 H-by-W-by-3 的 RGB 图像。');
    end

    positiveScalar = @(value) isnumeric(value) && isreal(value) && ...
        isscalar(value) && isfinite(value) && value > 0;
    nonnegativeScalar = @(value) isnumeric(value) && isreal(value) && ...
        isscalar(value) && isfinite(value) && value >= 0;
    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'Alpha', 0.15, nonnegativeScalar);
    addParameter(parser, 'Gamma', 0.8, positiveScalar);
    addParameter(parser, 'Sigma', 2, positiveScalar);
    addParameter(parser, 'Epsilon', 1e-3, positiveScalar);
    parse(parser, varargin{:});
    options = parser.Results;
    if options.Epsilon >= 1
        error('LIME:InvalidEpsilon', 'Epsilon 必须大于 0 且小于 1。');
    end

    rgb = double(rgb);
    initial = max(rgb, [], 3);
    [nRows, nCols] = size(initial);
    nPixels = numel(initial);

    if options.Alpha == 0 || nPixels == 1
        refined = initial;
    else
        gradientH = [diff(initial, 1, 2), zeros(nRows, 1)];
        gradientV = [diff(initial, 1, 1); zeros(1, nCols)];

        smoothH = gaussianSmoothReplicate(gradientH, options.Sigma);
        smoothV = gaussianSmoothReplicate(gradientV, options.Sigma);
        weightH = 1 ./ (abs(smoothH) + options.Epsilon);
        weightV = 1 ./ (abs(smoothV) + options.Epsilon);
        coefficientH = weightH ./ (abs(gradientH) + options.Epsilon);
        coefficientV = weightV ./ (abs(gradientV) + options.Epsilon);

        pixelIndex = reshape(1:nPixels, nRows, nCols);
        fromH = pixelIndex(:, 1:end-1);
        toH = pixelIndex(:, 2:end);
        fromV = pixelIndex(1:end-1, :);
        toV = pixelIndex(2:end, :);
        from = [fromH(:); fromV(:)];
        to = [toH(:); toV(:)];
        valuesH = coefficientH(:, 1:end-1);
        valuesV = coefficientV(1:end-1, :);
        values = options.Alpha * [valuesH(:); valuesV(:)];

        laplacian = sparse([from; to; from; to], [from; to; to; from], ...
            [values; values; -values; -values], nPixels, nPixels);
        systemMatrix = speye(nPixels) + laplacian;
        solution = systemMatrix \ initial(:);
        if any(~isfinite(solution))
            error('LIME:InvalidSolution', '光照求解产生非有限数值。');
        end
        refined = reshape(solution, nRows, nCols);
    end

    illumination = min(max(refined, options.Epsilon), 1) .^ options.Gamma;
    illumination = max(illumination, options.Epsilon);
    enhanced = min(max(bsxfun(@rdivide, rgb, illumination), 0), 1);
end

function result = gaussianSmoothReplicate(image, sigma)
    radius = ceil(3 * sigma);
    offsets = -radius:radius;
    kernel = exp(-(offsets .^ 2) / (2 * sigma ^ 2));
    kernel = kernel / sum(kernel);
    [nRows, nCols] = size(image);
    rows = min(nRows, max(1, (1-radius):(nRows+radius)));
    cols = min(nCols, max(1, (1-radius):(nCols+radius)));
    padded = image(rows, cols);
    result = conv2(kernel(:), kernel, padded, 'valid');
end
