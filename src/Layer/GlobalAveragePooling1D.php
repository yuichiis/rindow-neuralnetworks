<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling1D extends AbstractGlobalAveragePooling
{
    protected $rank = 1;

    public function __construct(object $backend,array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'globalaveragepooling1d');
        parent::__construct($backend, $leftargs);
    }
}
