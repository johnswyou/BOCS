% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

% function iSave1(fname, bayes, mle, hs, inputs_t)
% 	save(fname, 'bayes', 'mle', 'hs', 'inputs_t')
% end

function iSave1(fname, rnd, sa, bo, hs, inputs_t)
	save(fname, 'rnd', 'sa', 'bo', 'hs', 'inputs_t')
end
