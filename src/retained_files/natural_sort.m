function sorted = natural_sort(file_list)
% NATURAL_SORT sorts cell array of strings in natural (human-friendly) order.
%
% Usage: sorted = natural_sort(file_list)
%
% Example:
%   files = {'vol1.nii', 'vol10.nii', 'vol2.nii'};
%   sorted = natural_sort(files);
%
% Returns: {'vol1.nii', 'vol2.nii', 'vol10.nii'}

    % Extract tokens (numbers and text parts)
    tokens = regexp(file_list, '\d+|\D+', 'match');

    % Convert tokens to cell array of numeric or string
    maxTokens = max(cellfun(@length, tokens));
    numTokens = length(file_list);
    compMat = cell(numTokens, maxTokens);

    for i = 1:numTokens
        for j = 1:length(tokens{i})
            token = tokens{i}{j};
            numVal = str2double(token);
            if ~isnan(numVal)
                compMat{i, j} = numVal;
            else
                compMat{i, j} = token;
            end
        end
    end

    % Fill empty cells with empty strings for uniformity
    for i = 1:numTokens
        for j = length(tokens{i})+1:maxTokens
            compMat{i,j} = '';
        end
    end

    % Sort rows of compMat using a custom sort
    [~, sortIdx] = sortrows(compMat);

    sorted = file_list(sortIdx);
end
