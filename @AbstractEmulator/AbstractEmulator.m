classdef (Abstract) AbstractEmulator < handle
    methods (Abstract)
        [o,r,f] = start(varargins)
        [o,r,f] = reset(varargins)
        [o,r,f] = act(varargins)
    end 
end