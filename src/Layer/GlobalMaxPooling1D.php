<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalMaxPooling1D extends AbstractGlobalMaxPooling
{
    protected $rank = 1;

    public function __construct(object $backend,array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'globalmaxpooling1d');
        parent::__construct($backend, $leftargs);
    }
}
